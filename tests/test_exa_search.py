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
    assert result["backend"] == "exa"
    assert result["hits"][0]["doc_id"] == "exa-doc-1"
    assert result["hits"][0]["source"] == "info/news"
    assert result["hits"][0]["content"] == "Nvidia demand remained strong through the quarter."


def test_build_search_client_selects_exa_backend(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "secret")
    monkeypatch.setattr("tools.exa_search._load_exa_class", lambda: _FakeExa)

    client = build_search_client(backend="exa")

    assert isinstance(client, ExaSearchClient)
    assert client.health()["backend"] == "exa"
