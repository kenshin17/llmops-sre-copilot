from __future__ import annotations

import asyncio
from typing import Any

from pymilvus import Collection, connections
from opentelemetry import trace

from sre_copilot.config import Settings
from sre_copilot.services.embeddings import EmbeddingService
from sre_copilot.utils.logger import get_logger

logger = get_logger(__name__)


class MilvusRunbookStore:
    """
    Minimal Milvus search wrapper.
    Assumes a collection with vector field named 'vector' and text payload in 'text'.
    """

    def __init__(self, settings: Settings, embedder: EmbeddingService) -> None:
        self.settings = settings
        self.embedder = embedder
        self._connected = False
        self.tracer = trace.get_tracer("sre-copilot.milvus")

    def _connect(self) -> None:
        if self._connected:
            return
        connections.connect(
            alias="default",
            uri=self.settings.milvus_uri,
            user=self.settings.milvus_user,
            password=self.settings.milvus_password,
        )
        self._connected = True
        logger.info("Connected to Milvus at %s", self.settings.milvus_uri)

    async def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        try:
            with self.tracer.start_as_current_span("embedding.create") as span:
                span.set_attribute("provider", "openai")
                vector = await self.embedder.embed(query)
        except Exception as exc:  # pragma: no cover - network/service dependent
            logger.error("Embedding failed: %s", exc)
            return []

        try:
            self._connect()
            collection = Collection(self.settings.milvus_collection)
        except Exception as exc:  # pragma: no cover
            logger.error("Milvus connection failed: %s", exc)
            return []

        limit = top_k or self.settings.search_top_k
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

        def _do_search() -> list[dict[str, Any]]:
            results = collection.search(
                data=[vector],
                anns_field="vector",
                param=search_params,
                limit=limit,
                output_fields=["text"],
            )
            hits: list[dict[str, Any]] = []
            for hit in results[0]:
                hits.append(
                    {
                        "id": str(hit.id),
                        "score": float(hit.distance),
                        "text": hit.entity.get("text") if hit.entity else None,
                    }
                )
            return hits

        try:
            with self.tracer.start_as_current_span("milvus.search") as span:
                span.set_attribute("collection", self.settings.milvus_collection)
                span.set_attribute("limit", limit)
                hits = await asyncio.to_thread(_do_search)
                span.set_attribute("results.count", len(hits))
                return hits
        except Exception as exc:  # pragma: no cover
            logger.error("Milvus search failed: %s", exc)
            return []

    async def upsert_documents(self, docs: list[dict[str, Any]]) -> None:
        if not docs:
            return
        try:
            self._connect()
            collection = Collection(self.settings.milvus_collection)
        except Exception as exc:  # pragma: no cover
            logger.error("Milvus connection failed: %s", exc)
            raise

        def _do_insert() -> None:
            collection.insert(
                [
                    [doc["id"] for doc in docs],
                    [doc["text"] for doc in docs],
                    [doc["vector"] for doc in docs],
                ]
            )
            collection.flush()

        try:
            await asyncio.to_thread(_do_insert)
        except Exception as exc:  # pragma: no cover
            logger.error("Milvus insert failed: %s", exc)
            raise
