from __future__ import annotations

from typing import Any, Iterable, Mapping

import httpx
from fastapi import HTTPException
from openai import AsyncOpenAI, OpenAIError
from opentelemetry import trace
from sre_copilot.utils.logger import get_logger
from sre_copilot.observability.langfuse import LangfuseTracer, get_tracer

from sre_copilot.config import Settings
logger = get_logger(__name__)


Message = Mapping[str, str]


class LLMRouter:
    """
    Routes chat completion requests to OpenAI (primary) with Ollama fallback.
    """

    def __init__(
        self,
        settings: Settings,
        client: AsyncOpenAI | None = None,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        self.settings = settings
        self.openai = None
        if settings.openai_api_key:
            self.openai = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self.http = httpx.AsyncClient(timeout=20.0)
        self.tracer = tracer or get_tracer(settings)
        self.otel_tracer = trace.get_tracer("sre-copilot.llm")

    async def _call_openai(
        self, messages: Iterable[Message], model: str | None = None, **kwargs: Any
    ) -> str:
        response = await self.openai.chat.completions.create(
            model=model or self.settings.openai_model,
            messages=list(messages),
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    async def _call_ollama(
        self, messages: Iterable[Message], model: str | None = None, **kwargs: Any
    ) -> str:
        chat_model = model or self.settings.ollama_chat_model or self.settings.ollama_model
        payload = {
            "model": chat_model,
            "messages": list(messages),
            "stream": False,  # request single JSON object to avoid NDJSON decode errors
        }
        payload.update(kwargs)
        url = f"{self.settings.ollama_base_url}/api/chat"
        response = await self.http.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip()

    async def generate(
        self,
        messages: Iterable[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        use_fallback: bool = True,
    ) -> str:
        trace_span = self.tracer.start_span("llm.generate") if self.tracer else None
        with self.otel_tracer.start_as_current_span("llm.generate") as otel_span:
            if self.openai:
                try:
                    result = await self._call_openai(messages, model=model, temperature=temperature)
                    if trace_span:
                        trace_span.log({"backend": "openai", "model": model or self.settings.openai_model})
                        trace_span.end()
                    otel_span.set_attribute("backend", "openai")
                    otel_span.set_attribute("model", model or self.settings.openai_model)
                    return result
                except OpenAIError as exc:  # pragma: no cover - network handled at runtime
                    logger.warning("OpenAI call failed: %s", exc)
                    if not use_fallback:
                        if trace_span:
                            trace_span.log({"error": str(exc), "backend": "openai"})
                            trace_span.end()
                        otel_span.record_exception(exc)
                        otel_span.set_attribute("backend", "openai")
                        raise HTTPException(status_code=502, detail="Primary LLM unavailable") from exc
                except Exception as exc:  # pragma: no cover
                    logger.error("Unexpected OpenAI error: %s", exc)
                    if not use_fallback:
                        if trace_span:
                            trace_span.log({"error": str(exc), "backend": "openai"})
                            trace_span.end()
                        otel_span.record_exception(exc)
                        otel_span.set_attribute("backend", "openai")
                        raise HTTPException(status_code=502, detail="Primary LLM unavailable") from exc

            try:
                result = await self._call_ollama(messages, model=model, temperature=temperature)
                if trace_span:
                    trace_span.log({"backend": "ollama", "model": model or self.settings.ollama_chat_model or self.settings.ollama_model})
                    trace_span.end()
                otel_span.set_attribute("backend", "ollama")
                otel_span.set_attribute("model", model or self.settings.ollama_chat_model or self.settings.ollama_model)
                return result
            except Exception as exc:  # pragma: no cover
                logger.error("Fallback Ollama failed: %s", exc)
                if trace_span:
                    trace_span.log({"error": str(exc), "backend": "ollama"})
                    trace_span.end()
                otel_span.record_exception(exc)
                otel_span.set_attribute("backend", "ollama")
                raise HTTPException(status_code=502, detail="LLM backends unavailable") from exc
