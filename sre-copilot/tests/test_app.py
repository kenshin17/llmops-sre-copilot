from fastapi.testclient import TestClient

from sre_copilot.app import app


def test_health_ready():
    client = TestClient(app)
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
