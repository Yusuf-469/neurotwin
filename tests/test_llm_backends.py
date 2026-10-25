"""LLM backends: provider-neutral client, API routing, cache and cost cap (no network)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from neurotwin.api import app as app_module
from neurotwin.explanation.openai_compatible import OpenAICompatibleClient


class _FakeProvider(BaseHTTPRequestHandler):
    calls = 0
    reply: dict = {"choices": [{"message": {"content": "The policy placed a priority 3 patient."}}]}

    def do_POST(self):  # noqa: N802 (http.server naming)
        type(self).calls += 1
        length = int(self.headers["Content-Length"])
        json.loads(self.rfile.read(length))
        assert self.headers["Authorization"] == "Bearer test-key"
        body = json.dumps(type(self).reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture()
def provider():
    server = HTTPServer(("127.0.0.1", 0), _FakeProvider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _FakeProvider.calls = 0
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def test_openai_compatible_client_happy_path_and_bad_format(provider):
    client = OpenAICompatibleClient(provider, "free-model", "test-key")
    assert client.complete("hi") == "The policy placed a priority 3 patient."
    old = _FakeProvider.reply
    _FakeProvider.reply = {"unexpected": True}
    try:
        with pytest.raises(RuntimeError):
            client.complete("hi")
    finally:
        _FakeProvider.reply = old


def test_api_uses_configured_free_llm_with_cache_and_hourly_cap(provider, monkeypatch):
    monkeypatch.setenv("NEUROTWIN_LLM_BASE_URL", provider)
    monkeypatch.setenv("NEUROTWIN_LLM_MODEL", "free-model")
    monkeypatch.setenv("NEUROTWIN_LLM_API_KEY", "test-key")
    monkeypatch.setattr(app_module, "LLM_MAX_PER_HOUR", 1)
    app_module._LLM_CALLS.clear()
    app_module._LLM_CACHE.clear()
    client = TestClient(app_module.app)
    assert client.get("/api/health").json()["llm_explanations"] is True
    sid = client.post("/api/simulation/start", json={"scenario": "A", "seed": 2}).json()[
        "session_id"
    ]
    client.post(f"/api/simulation/{sid}/step", json={"hours": 24})

    first = client.get(f"/api/simulation/{sid}/explanations/0").json()["explanation"]
    assert first["generated_by"] == "llm" and _FakeProvider.calls == 1
    again = client.get(f"/api/simulation/{sid}/explanations/0").json()["explanation"]
    assert again == first and _FakeProvider.calls == 1  # cached: no second paid call
    capped = client.get(f"/api/simulation/{sid}/explanations/1").json()["explanation"]
    assert capped["generated_by"] == "template" and _FakeProvider.calls == 1
    assert "limit" in capped["verification_issues"][0]


def test_without_configuration_the_template_is_used(monkeypatch):
    for k in (
        "NEUROTWIN_LLM_BASE_URL",
        "NEUROTWIN_LLM_MODEL",
        "NEUROTWIN_LLM_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    client = TestClient(app_module.app)
    assert client.get("/api/health").json()["llm_explanations"] is False
    sid = client.post("/api/simulation/start", json={"scenario": "A", "seed": 2}).json()[
        "session_id"
    ]
    client.post(f"/api/simulation/{sid}/step", json={"hours": 24})
    exp = client.get(f"/api/simulation/{sid}/explanations/0").json()["explanation"]
    assert exp["generated_by"] == "template"
