from fastapi.testclient import TestClient

from sre_copilot.app import app
from sre_copilot.config import get_settings


def override_settings():
    settings = get_settings()
    settings.api_keys = ["test-key"]
    settings.rate_limit_per_minute = 1
    settings.redis_url = None
    return settings


def setup_module(_module):
    app.dependency_overrides[get_settings] = override_settings


def teardown_module(_module):
    app.dependency_overrides.pop(get_settings, None)


def test_missing_api_key_rejected():
    client = TestClient(app)
    resp = client.post("/v1/search", json={"query": "hi"})
    assert resp.status_code == 401


def test_rate_limit_enforced():
    client = TestClient(app)
    headers = {"x-api-key": "test-key"}
    first = client.post("/v1/search", json={"query": "hi"}, headers=headers)
    assert first.status_code in (200, 404, 500, 502)  # allow dependency failures
    second = client.post("/v1/search", json={"query": "hi"}, headers=headers)
    assert second.status_code == 429
