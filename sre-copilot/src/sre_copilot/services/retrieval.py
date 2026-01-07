from __future__ import annotations

from typing import Any

from fastapi import Depends
from opentelemetry import trace

from sre_copilot.config import Settings, get_settings
from sre_copilot.guardrails import GuardrailsEngine
from sre_copilot.services.embeddings import EmbeddingService
from sre_copilot.services.llm_router import LLMRouter
from sre_copilot.services.milvus_client import MilvusRunbookStore
from sre_copilot.services.redis_cache import RedisCache
from sre_copilot.utils.logger import get_logger


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        guardrails: GuardrailsEngine | None = None,
        llm_router: LLMRouter | None = None,
        store: MilvusRunbookStore | None = None,
        cache: RedisCache | None = None,
    ) -> None:
        self.settings = settings
        self.guardrails = guardrails or GuardrailsEngine(settings)
        embedder = EmbeddingService(settings)
        self.store = store or MilvusRunbookStore(settings, embedder)
        self.llm_router = llm_router or LLMRouter(settings)
        self.cache = cache
        if not self.cache and self.settings.cache_enabled and self.settings.redis_url:
            self.cache = RedisCache(settings)
        self.tracer = trace.get_tracer("sre-copilot.retrieval")
        self.logger = get_logger(__name__)
        # Prefer OpenAI model when an API key is configured; otherwise use Ollama chat/ollama_model
        if settings.openai_api_key:
            self.chat_model = settings.openai_model
        else:
            self.chat_model = settings.ollama_chat_model or settings.ollama_model

    async def search_runbooks(self, query: str) -> dict[str, Any]:
        with self.tracer.start_as_current_span("guardrails.validate") as span:
            guardrail_result = await self.guardrails.validate(query)
            span.set_attribute("prompt_injection", guardrail_result.prompt_injection)
            if guardrail_result.pii_matches:
                span.set_attribute("pii_matches.count", len(guardrail_result.pii_matches))
            span.set_attribute("blocked", guardrail_result.blocked)
        self.logger.debug(
            "Guardrails result blocked=%s prompt_injection=%s pii=%d",
            guardrail_result.blocked,
            guardrail_result.prompt_injection,
            len(guardrail_result.pii_matches),
        )
        if guardrail_result.blocked:
            return {
                "query": guardrail_result.sanitized_text,
                "prompt_injection": True,
                "blocked": True,
                "pii_matches": guardrail_result.pii_matches,
                "results": [],
                "reason": guardrail_result.reason,
            }

        rewritten = guardrail_result.sanitized_text
        if self.settings.query_rewrite_enabled:
            rewritten = await self._rewrite_query(guardrail_result.sanitized_text)

        cache_key = None
        if self.cache:
            cache_key = self.cache.cache_key(rewritten, self.settings.search_top_k)
            cached = await self.cache.get(cache_key)
            if cached:
                self.logger.debug("Cache hit for query key=%s", cache_key)
                return cached

        with self.tracer.start_as_current_span("milvus.search") as span:
            span.set_attribute("top_k", self.settings.search_top_k)
            span.set_attribute("query.original", guardrail_result.sanitized_text)
            span.set_attribute("query.rewritten", rewritten)
            hits = await self.store.search(rewritten)
            span.set_attribute("results.count", len(hits))
        self.logger.debug(
            "Search completed for query=%s rewritten=%s results=%d",
            guardrail_result.sanitized_text,
            rewritten,
            len(hits),
        )

        response = {
            "query": rewritten,
            "original_query": guardrail_result.sanitized_text,
            "prompt_injection": guardrail_result.prompt_injection,
            "pii_matches": guardrail_result.pii_matches,
            "blocked": False,
            "results": hits,
        }
        if self.cache and cache_key:
            await self.cache.set(cache_key, response)
        return response

    async def answer(self, query: str) -> dict[str, Any]:
        # Retrieve context then call LLM router for a grounded response.
        search_result = await self.search_runbooks(query)
        if search_result.get("blocked") or search_result.get("prompt_injection"):
            search_result["answer"] = "Blocked: sensitive or disallowed content detected."
            return search_result

        context_chunks = [hit.get("text", "") for hit in search_result["results"] if hit.get("text")]
        messages = [
            {
                "role": "system",
                "content": "You are an SRE copilot. Use provided runbook snippets to answer.",
            },
            {"role": "user", "content": f"Question: {search_result['query']}"},
        ]
        if context_chunks:
            context = "\n---\n".join(context_chunks)
            messages.append({"role": "system", "content": f"Context:\n{context}"})
        with self.tracer.start_as_current_span("llm.generate") as span:
            span.set_attribute("context.count", len(context_chunks))
            answer = await self.llm_router.generate(messages)
        return {"answer": answer, **search_result}

    async def _rewrite_query(self, text: str) -> str:
        # If no chat backend, fall back to original.
        if not self.llm_router:
            return text
        system_prompt = (
            "Rewrite the user's query to be concise and specific for log/runbook search. "
            "Do not add new facts. Do not change intent. Return only the rewritten query."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        try:
            return await self.llm_router.generate(messages, model=self.chat_model or None, temperature=0.0)
        except Exception:
            return text


def get_retrieval_service(settings: Settings = Depends(get_settings)) -> RetrievalService:
    return RetrievalService(settings)
