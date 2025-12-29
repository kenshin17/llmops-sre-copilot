from fastapi.testclient import TestClient

from sre_copilot.app import app
from sre_copilot.services.retrieval import RetrievalService, get_retrieval_service


class FakeRetrievalService:
    async def search_runbooks(self, query: str):
        return {
            "query": query,
            "prompt_injection": False,
            "pii_matches": [],
            "results": [
                {"id": "1", "score": 0.9, "text": "restart the service"},
                {"id": "2", "score": 0.8, "text": "check logs"},
            ],
        }

    async def answer(self, query: str):
        base = await self.search_runbooks(query)
        return {"answer": "mocked answer", **base}


def override_retrieval_service() -> RetrievalService:  # pragma: no cover - used via override
    return FakeRetrievalService()  # type: ignore[return-value]


def setup_module(_module):
    app.dependency_overrides[get_retrieval_service] = override_retrieval_service


def teardown_module(_module):
    app.dependency_overrides.pop(get_retrieval_service, None)


def test_search_endpoint_returns_hits():
    client = TestClient(app)
    resp = client.post("/v1/search", json={"query": "incident?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"]
    assert body["results"][0]["text"] == "restart the service"


def test_answer_endpoint_returns_answer_and_results():
    client = TestClient(app)
    resp = client.post("/v1/answer", json={"query": "why is pod down"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "mocked answer"
    assert len(body["results"]) == 2
