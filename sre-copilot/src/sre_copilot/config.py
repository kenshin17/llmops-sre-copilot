from functools import lru_cache
import json
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SRE_", extra="ignore")

    api_keys: list[str] | str | None = Field(default=None)
    api_key_header: str = "x-api-key"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    guardrails_enabled: bool = True
    guardrails_model: str | None = None
    guardrails_base_url: str | None = None
    ollama_base_url: str | None = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_embed_model: str | None = None
    ollama_chat_model: str | None = None

    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "runbooks"
    milvus_user: str | None = None
    milvus_password: str | None = None
    search_top_k: int = 3

    redis_url: str | None = None

    langfuse_base_url: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_public_key: str | None = None

    otel_enabled: bool = False
    otel_endpoint: str | None = "http://localhost:4317"
    otel_service_name: str = "sre-copilot"
    otel_excluded_urls: str = r"^/health$,^/ready$"
    otel_metrics_enabled: bool = False

    loki_url: str | None = None
    loki_query: str = ""
    loki_tenant: str | None = None
    prom_url: str | None = None
    prom_query: str = 'up'
    tempo_url: str | None = None
    tempo_service: str | None = None
    ingest_lookback_minutes: int = 120
    query_rewrite_enabled: bool = False
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    http_timeout: float = 30.0

    @field_validator("api_keys", mode="before")
    @classmethod
    def split_api_keys(cls, value: Any) -> list[str] | str | None:
        # Let model_validator normalize; just return raw here to avoid JSON decode failures.
        return value

    @model_validator(mode="after")
    def normalize_api_keys(self) -> "Settings":
        raw = self.api_keys
        if raw is None or raw == "":
            self.api_keys = []
            return self
        if isinstance(raw, str):
            text = raw.strip()
            if text.startswith("["):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        self.api_keys = [str(item).strip() for item in parsed if str(item).strip()]
                        return self
                except Exception:
                    pass
            self.api_keys = [item.strip() for item in text.split(",") if item.strip()]
            return self
        if isinstance(raw, list):
            self.api_keys = [str(item).strip() for item in raw if str(item).strip()]
            return self
        self.api_keys = []
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
