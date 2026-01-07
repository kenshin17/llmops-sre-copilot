import pytest

from sre_copilot.config import Settings
from sre_copilot.guardrails import GuardrailResult
from sre_copilot.services.llm_router import LLMRouter
from sre_copilot.services.retrieval import RetrievalService


class FakeGuardrails:
    async def validate(self, user_input: str):
        return GuardrailResult(
            sanitized_text=user_input,
            pii_matches=[],
            prompt_injection=False,
            blocked=False,
            reason=None,
        )


class FakeStore:
    async def search(self, query: str):
        return [
            {"id": "1", "score": 0.9, "text": "restart the service"},
            {"id": "2", "score": 0.8, "text": "check logs"},
        ]


class FakeRouter(LLMRouter):
    def __init__(self):
        pass

    async def generate(self, messages, **_kwargs):
        # Echo back the last user message for observability in tests.
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        return f"answer to: {user_msgs[-1]}"


@pytest.mark.asyncio
async def test_retrieval_service_answer_uses_store_and_router():
    svc = RetrievalService(
        settings=Settings(openai_api_key="dummy", api_keys=[]),
        guardrails=FakeGuardrails(),
        llm_router=FakeRouter(),
        store=FakeStore(),
    )

    result = await svc.answer("service is down")
    assert result["results"][0]["text"] == "restart the service"
    assert "answer to" in result["answer"]


@pytest.mark.asyncio
async def test_retrieval_service_search_only_returns_hits():
    svc = RetrievalService(
        settings=Settings(openai_api_key="dummy", api_keys=[]),
        guardrails=FakeGuardrails(),
        llm_router=FakeRouter(),
        store=FakeStore(),
    )
    result = await svc.search_runbooks("check logs")
    assert len(result["results"]) == 2


class FakeCache:
    def __init__(self):
        self.store = {}

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value, ttl_seconds=None):
        self.store[key] = value
        self.ttl = ttl_seconds


class CountingStore(FakeStore):
    def __init__(self):
        super().__init__()
        self.calls = 0

    async def search(self, query: str):
        self.calls += 1
        return await super().search(query)


@pytest.mark.asyncio
async def test_retrieval_service_uses_cache_when_available():
    cache = FakeCache()
    store = CountingStore()
    svc = RetrievalService(
        settings=Settings(openai_api_key="dummy", api_keys=[]),
        guardrails=FakeGuardrails(),
        llm_router=FakeRouter(),
        store=store,
        cache=cache,
    )

    first = await svc.search_runbooks("restart api service")
    assert store.calls == 1

    second = await svc.search_runbooks("restart api service")
    assert store.calls == 1  # cache hit should prevent second search
    assert second == first
