import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sre_copilot.config import Settings
from sre_copilot.guardrails.injection import detect_injection
from sre_copilot.guardrails.pii import find_pii, mask_matches
from sre_copilot.utils.logger import get_logger


@dataclass
class GuardrailResult:
    sanitized_text: str
    pii_matches: list[str]
    prompt_injection: bool
    blocked: bool
    reason: str | None = None


class GuardrailsEngine:
    """
    Guardrails entrypoint. Uses NeMo Guardrails when available/configured and
    falls back to lightweight heuristics otherwise.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        rails: Any | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.logger = get_logger(__name__)
        self._rails = rails or self._build_nemo_rails()

    def _build_nemo_rails(self) -> Any | None:
        """Initialize NeMo Guardrails if installed and enabled."""
        if not self.settings.guardrails_enabled:
            return None
        if not self.settings.openai_api_key and not self.settings.guardrails_base_url:
            self.logger.info("guardrails enabled but no OpenAI credentials; using heuristics")
            return None
        try:
            from nemoguardrails import LLMRails, RailsConfig
        except Exception as exc:  # pragma: no cover - optional dependency
            self.logger.info("nemoguardrails not available, using heuristics: %s", exc)
            return None

        config_dir = Path(__file__).parent / "config"
        if not config_dir.exists():
            self.logger.warning("Guardrails config missing at %s", config_dir)
            return None

        if self.settings.openai_api_key:
            os.environ.setdefault("OPENAI_API_KEY", self.settings.openai_api_key)

        model_name = self.settings.guardrails_model or self.settings.openai_model
        config = RailsConfig.from_path(str(config_dir))
        for model_cfg in config.models:
            if getattr(model_cfg, "type", "") == "main":
                if model_name:
                    model_cfg.model = model_name
                if self.settings.guardrails_base_url:
                    model_cfg.parameters["base_url"] = self.settings.guardrails_base_url

        self.logger.debug("Initialized NeMo guardrails with model=%s base_url=%s", model_name,
                          self.settings.guardrails_base_url or "default")
        return LLMRails(config)

    async def validate(self, user_input: str) -> GuardrailResult:
        self.logger.debug("Guardrails validate start")
        # Always compute heuristic result for fallback/telemetry purposes.
        heuristic_result = self._heuristic_validate(user_input)
        if not self._rails:
            self.logger.debug("Guardrails using heuristic result (NeMo disabled/unavailable)")
            return heuristic_result

        try:
            nemo_result = await self._run_nemo_guard(user_input)
            parsed = self._parse_nemo_result(nemo_result, heuristic_result)
            return parsed
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("NeMo guardrails failed, using heuristics: %s", exc)
            return heuristic_result

    async def _run_nemo_guard(self, user_input: str) -> Any:
        messages = [{"role": "user", "content": user_input}]
        # LLMRails.generate is synchronous; run it off the event loop.
        self.logger.debug("Invoking NeMo guardrails model")
        return await asyncio.to_thread(self._rails.generate, messages=messages)

    def _parse_nemo_result(
        self, nemo_result: Any, fallback: GuardrailResult
    ) -> GuardrailResult:
        """Parse NeMo guardrails JSON decision; fall back if parsing fails."""
        payload: dict[str, Any]
        if isinstance(nemo_result, dict):
            payload = nemo_result
        else:
            try:
                payload = json.loads(str(nemo_result))
            except Exception:
                self.logger.warning("Unable to parse guardrails output: %s", nemo_result)
                return fallback

        sanitized = (
            payload.get("sanitized_text") or payload.get("sanitized") or fallback.sanitized_text
        )
        pii_matches = payload.get("pii") or payload.get("pii_matches") or fallback.pii_matches
        prompt_injection = bool(
            payload.get("prompt_injection")
            or payload.get("promptInjection")
            or fallback.prompt_injection
        )
        blocked = bool(payload.get("blocked", fallback.blocked))
        reason = payload.get("reason") or fallback.reason
        if blocked and not reason:
            reason = "blocked by guardrails"

        self.logger.debug(
            "Guardrails decision: blocked=%s prompt_injection=%s pii_matches=%d reason=%s",
            blocked,
            prompt_injection,
            len(pii_matches),
            reason,
        )
        return GuardrailResult(
            sanitized_text=sanitized,
            pii_matches=list(pii_matches),
            prompt_injection=prompt_injection,
            blocked=blocked,
            reason=reason,
        )

    def _heuristic_validate(self, user_input: str) -> GuardrailResult:
        pii_hits = find_pii(user_input)
        sanitized = mask_matches(user_input, pii_hits)
        injection_flag = detect_injection(user_input)
        reason = None
        blocked = False
        if injection_flag:
            reason = "prompt injection detected"
            blocked = True
        elif pii_hits:
            reason = "PII or sensitive token detected"
            blocked = True
        return GuardrailResult(
            sanitized_text=sanitized,
            pii_matches=pii_hits,
            prompt_injection=injection_flag,
            blocked=blocked,
            reason=reason,
        )
