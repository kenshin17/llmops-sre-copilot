from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sre_copilot.config import Settings
from sre_copilot.utils.logger import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    from langfuse import Langfuse
except Exception:  # pragma: no cover
    Langfuse = None  # type: ignore[misc]


@dataclass
class LangfuseSpan:
    name: str
    trace: Any | None

    def log(self, payload: dict[str, Any]) -> None:
        if self.trace:
            try:
                self.trace.event(name="log", input=payload)
            except Exception as exc:  # pragma: no cover
                logger.warning("Langfuse event failed: %s", exc)
        logger.info("[trace] %s %s", self.name, payload)

    def end(self) -> None:
        if self.trace:
            try:
                self.trace.end()
            except Exception as exc:  # pragma: no cover
                logger.warning("Langfuse end failed: %s", exc)
        logger.info("[trace] %s end", self.name)


class LangfuseTracer:
    def __init__(self, client: Any | None) -> None:
        self.client = client

    def start_span(self, name: str) -> LangfuseSpan:
        trace = None
        if self.client:
            try:
                trace = self.client.trace(name=name)
            except Exception as exc:  # pragma: no cover
                logger.warning("Langfuse trace start failed: %s", exc)
        return LangfuseSpan(name=name, trace=trace)


def get_tracer(settings: Settings) -> Optional[LangfuseTracer]:
    if not (settings.langfuse_secret_key and settings.langfuse_public_key and settings.langfuse_base_url):
        return None
    if Langfuse is None:
        logger.warning("Langfuse dependency missing; install langfuse to enable tracing.")
        return None
    try:
        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
        )
        logger.info("Langfuse tracer initialized at %s", settings.langfuse_base_url)
        return LangfuseTracer(client)
    except Exception as exc:  # pragma: no cover
        logger.warning("Langfuse client initialization failed: %s", exc)
        return None
