from __future__ import annotations

from types import SimpleNamespace

from tools.exa_search import ExaSearchClient
from tools.search_clients import build_search_client


class _FakeExa:
    def __init__(self, api_key, base_url="https://api.exa.ai", user_agent=None):
        self.api_key = api_key
        self.base_url = base_url
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, "kwargs": dict(kwargs)})
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    id="exa-doc-1",
                    title="Nvidia supply chain update",
                    url="https://example.com/nvda",
                    score=0.91,
                    published_date="2026-03-06T10:00:00.000Z",
                    text="Nvidia demand remained strong through the quarter.",
                    summary=None,
                    highlights=None,
                )
            ]
        )


class _FailingExa:
    def __init__(self, api_key, base_url="https://api.exa.ai", user_agent=None):
        self.api_key = api_key
        self.base_url = base_url

    def search(self, query, **kwargs):
        raise RuntimeError("proxy connection failed")


def test_exa_search_client_maps_response(monkeypatch):
    fake = _FakeExa("secret")
    monkeypatch.setenv("EXA_API_KEY", "secret")
    monkeypatch.setattr("tools.exa_search._load_exa_class", lambda: lambda api_key, base_url="https://api.exa.ai", user_agent=None: fake)

    client = ExaSearchClient()
    result = client.search(
        "Latest news on Nvidia",
        time="2026-03-06T15:59:59.999Z",
        source="news",
        limit=3,
    )

    assert fake.calls[0]["query"] == "Latest news on Nvidia"
    assert fake.calls[0]["kwargs"]["end_published_date"] == "2026-03-06T15:59:59.999Z"
    assert fake.calls[0]["kwargs"]["category"] == "news"
    assert fake.calls[0]["kwargs"]["contents"]["text"]["max_characters"] == 4000
    assert result["backend"] == "exa"
    assert result["hits"][0]["doc_id"] == "exa-doc-1"
    assert result["hits"][0]["source"] == "info/news"
    assert result["hits"][0]["content"] == "Nvidia supply chain update\nNvidia demand remained strong through the quarter."


def test_build_search_client_selects_exa_backend(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "secret")
    monkeypatch.setattr("tools.exa_search._load_exa_class", lambda: _FakeExa)

    client = build_search_client(backend="exa")

    assert isinstance(client, ExaSearchClient)
    assert client.health()["backend"] == "exa"


def test_build_search_client_does_not_reuse_local_search_base_for_exa(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "secret")
    monkeypatch.delenv("EXA_BASE_URL", raising=False)
    monkeypatch.setattr("tools.exa_search._load_exa_class", lambda: _FakeExa)

    client = build_search_client(
        backend="exa",
        base_url="http://127.0.0.1:8000",
    )

    assert isinstance(client, ExaSearchClient)
    assert client.base_url == "https://api.exa.ai"


def test_build_search_client_rejects_local_retrieval_mode_for_exa(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "secret")
    monkeypatch.setattr("tools.exa_search._load_exa_class", lambda: _FakeExa)

    try:
        build_search_client(backend="exa", default_mode="hybrid")
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected build_search_client() to reject local retrieval modes for exa")

    assert "search_retrieval_mode applies only to the local search backend" in message


def test_exa_search_client_surfaces_proxy_context(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "secret")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:7897")
    monkeypatch.setattr("tools.exa_search._load_exa_class", lambda: _FailingExa)

    client = ExaSearchClient()

    try:
        client.search("Latest news on Nvidia")
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ExaSearchClient.search() to raise")

    assert "Exa search request failed" in message
    assert "https_proxy=http://127.0.0.1:7897" in message
