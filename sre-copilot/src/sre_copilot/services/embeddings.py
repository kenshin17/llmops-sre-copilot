from __future__ import annotations

from typing import Iterable, List
import hashlib
import itertools
import httpx

from openai import AsyncOpenAI

from sre_copilot.config import Settings
from sre_copilot.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        self.settings = settings
        self.client = None
        self.provider = "hash"
        self.dim = 1536

        if settings.openai_api_key:
            self.client = client or AsyncOpenAI(api_key=settings.openai_api_key)
            self.provider = "openai"
        elif settings.ollama_base_url:
            self.provider = "ollama"

        self.model = "text-embedding-3-small"
        self.http_client = httpx.AsyncClient(timeout=settings.http_timeout)

    async def embed(self, text: str) -> List[float]:
        model = self.settings.ollama_embed_model or self.settings.ollama_model

        if self.provider == "openai" and self.client:
            response = await self.client.embeddings.create(model=self.model, input=[text])
            return self._normalize(list(response.data[0].embedding))

        if self.provider == "ollama" and self.settings.ollama_base_url:
            try:
                resp = await self.http_client.post(
                    f"{self.settings.ollama_base_url}/api/embeddings",
                    json={"model": model, "prompt": text},
                )
                resp.raise_for_status()
                data = resp.json()
                vec = data.get("embedding") or data.get("data") or []
                return self._normalize(vec)
            except Exception as exc:
                # fall back to hash below
                logger.warning("Ollama embedding failed, falling back to hash: %s", exc)

        # Fallback deterministic hash embedding
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return self._normalize([b / 255 for b in digest])

    async def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        texts_list = list(texts)

        if self.provider == "openai" and self.client:
            response = await self.client.embeddings.create(model=self.model, input=texts_list)
            return [self._normalize(list(item.embedding)) for item in response.data]

        if self.provider == "ollama" and self.settings.ollama_base_url:
            model = self.settings.ollama_embed_model or self.settings.ollama_model
            results: list[list[float]] = []
            try:
                for text in texts_list:
                    resp = await self.http_client.post(
                        f"{self.settings.ollama_base_url}/api/embeddings",
                        json={"model": model, "prompt": text},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    vec = data.get("embedding") or data.get("data") or []
                    results.append(self._normalize(vec))
                return results
            except Exception as exc:
                logger.warning("Ollama batch embedding failed, falling back to hash: %s", exc)

        # Fallback deterministic hash embeddings
        vectors: list[list[float]] = []
        for text in texts_list:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append(self._normalize([b / 255 for b in digest]))
        return vectors

    def _normalize(self, vec: list[float]) -> list[float]:
        if not vec:
            return [0.0] * self.dim
        if len(vec) == self.dim:
            return vec
        if len(vec) > self.dim:
            return vec[: self.dim]
        # pad by cycling values
        cycle = itertools.cycle(vec)
        return [next(cycle) for _ in range(self.dim)]
