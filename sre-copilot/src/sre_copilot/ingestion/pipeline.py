from __future__ import annotations

import asyncio
import uuid
from typing import Iterable, List

from sre_copilot.config import Settings, get_settings
from sre_copilot.ingestion.sources import fetch_loki_logs, fetch_prometheus_metrics, fetch_tempo_traces
from sre_copilot.services.embeddings import EmbeddingService
from sre_copilot.services.milvus_client import MilvusRunbookStore
from sre_copilot.services.milvus_schema import ensure_collection
from sre_copilot.utils.logger import get_logger

logger = get_logger(__name__)


def chunk_text(text: str, max_len: int = 500) -> List[str]:
    words = text.split()
    chunks: list[list[str]] = []
    current: list[str] = []
    total = 0
    for word in words:
        if total + len(word) + 1 > max_len and current:
            chunks.append(current)
            current = []
            total = 0
        current.append(word)
        total += len(word) + 1
    if current:
        chunks.append(current)
    return [" ".join(chunk) for chunk in chunks]


async def ingest_observability(settings: Settings) -> int:
    embedder = EmbeddingService(settings)
    store = MilvusRunbookStore(settings, embedder)

    await ensure_collection(settings)

    # Fetch from sources
    loki_docs, prom_docs, tempo_docs = await asyncio.gather(
        fetch_loki_logs(settings), fetch_prometheus_metrics(settings), fetch_tempo_traces(settings)
    )
    raw_docs = loki_docs + prom_docs + tempo_docs
    if not raw_docs:
        logger.info("No documents fetched from observability sources.")
        return 0

    # Chunk and embed
    to_write = []
    for doc in raw_docs:
        for chunk in chunk_text(doc):
            to_write.append({"id": str(uuid.uuid4()), "text": chunk})

    embeddings = await embedder.embed_batch([item["text"] for item in to_write])
    for item, vector in zip(to_write, embeddings):
        item["vector"] = vector

    await store.upsert_documents(to_write)
    logger.info("Ingested %s chunks into Milvus collection %s", len(to_write), settings.milvus_collection)
    return len(to_write)


def run_ingest_sync() -> None:
    settings = get_settings()
    asyncio.run(ingest_observability(settings))


if __name__ == "__main__":
    run_ingest_sync()
