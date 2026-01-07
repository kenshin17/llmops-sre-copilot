import pytest

from sre_copilot.config import Settings
from sre_copilot.services.llm_router import LLMRouter


class RouterSpy(LLMRouter):
    def __init__(self):
        self.openai_called = False
        self.ollama_called = False
        super().__init__(settings=Settings(openai_api_key="dummy"))

    async def _call_openai(self, messages, model=None, **kwargs):
        self.openai_called = True
        return "primary response"

    async def _call_ollama(self, messages, model=None, **kwargs):
        self.ollama_called = True
        return "fallback response"


class RouterFailPrimary(RouterSpy):
    async def _call_openai(self, messages, model=None, **kwargs):
        self.openai_called = True
        raise RuntimeError("primary down")


@pytest.mark.asyncio
async def test_llm_router_primary_succeeds():
    router = RouterSpy()
    result = await router.generate([{"role": "user", "content": "hi"}])
    assert result == "primary response"
    assert router.openai_called is True
    assert router.ollama_called is False


@pytest.mark.asyncio
async def test_llm_router_fallback_on_error():
    router = RouterFailPrimary()
    result = await router.generate([{"role": "user", "content": "hi"}])
    assert result == "fallback response"
    assert router.openai_called is True
    assert router.ollama_called is True
