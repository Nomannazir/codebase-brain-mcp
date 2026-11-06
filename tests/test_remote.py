import pytest
from starlette.testclient import TestClient

from codebase_brain_mcp.remote import RemoteConfig, RemoteConfigError, build_app
from codebase_brain_mcp.server import build_server

TOKEN = "t" * 32
INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}},
}
HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


def client(brain, **config):
    app = build_app(build_server(brain), RemoteConfig(**{"host": "0.0.0.0", "port": 8765, "token": TOKEN, **config}))
    return TestClient(app, base_url="http://brain.example.com")


def test_public_host_requires_a_strong_token():
    with pytest.raises(RemoteConfigError, match="without CODEBASE_BRAIN_TOKEN"):
        RemoteConfig(host="0.0.0.0", port=1, token=None).validate()
    with pytest.raises(RemoteConfigError, match="at least"):
        RemoteConfig(host="0.0.0.0", port=1, token="short").validate()
    RemoteConfig(host="127.0.0.1", port=1, token=None).validate()  # local use needs no token


def test_health_check_is_open(brain):
    with client(brain) as c:
        response = c.get("/healthz")
    assert response.status_code == 200 and response.json() == {"status": "ok"}


@pytest.mark.parametrize("authorization", [None, "Bearer wrong-token-wrong-token-wrong", f"Basic {TOKEN}", TOKEN])
def test_mcp_rejects_missing_or_wrong_tokens(brain, authorization):
    headers = {**HEADERS, **({"authorization": authorization} if authorization else {})}
    with client(brain) as c:
        response = c.post("/mcp", json=INIT, headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")


def test_mcp_accepts_the_right_token(brain):
    with client(brain) as c:
        response = c.post("/mcp", json=INIT, headers={**HEADERS, "authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 200
    assert "codebase-brain" in response.text


def test_loopback_server_blocks_dns_rebinding(brain):
    app = build_app(build_server(brain), RemoteConfig(host="127.0.0.1", port=8765, token=None))
    with TestClient(app, base_url="http://evil.example.com") as c:
        response = c.post("/mcp", json=INIT, headers=HEADERS)
    assert response.status_code in (403, 421)


def test_remote_mode_refuses_local_paths_and_huge_repos(settings, sample_repo):
    import dataclasses

    from codebase_brain_mcp.brain import Brain, BrainError
    from codebase_brain_mcp.embeddings import HashEmbedder

    remote = Brain(dataclasses.replace(settings, allow_local_paths=False), HashEmbedder())
    with pytest.raises(BrainError, match="only indexes public https git URLs"):
        remote.index(str(sample_repo))

    tiny = Brain(dataclasses.replace(settings, max_files=1), HashEmbedder())
    with pytest.raises(BrainError, match="over this server's limit"):
        tiny.index(str(sample_repo))
