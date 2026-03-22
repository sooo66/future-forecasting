from __future__ import annotations

import json
import re

from fastapi.testclient import TestClient

from tools.corpus import build_corpus
from tools.search import SearchClient, SearchEngine, build_bm25_index, create_app


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


class _FakeHit:
    def __init__(self, docid: str, score: float) -> None:
        self.docid = docid
        self.score = score


class _FakeSearcher:
    def __init__(self, corpus_path):
        self.rows = []
        with corpus_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                self.rows.append(json.loads(line))

    def search(self, query, k):
        tokens = [token for token in re.findall(r"\w+", str(query).lower()) if token]
        scored = []
        for row in self.rows:
            contents = str(row.get("contents") or "").lower()
            score = float(sum(contents.count(token) for token in tokens))
            if score <= 0:
                continue
            scored.append(_FakeHit(str(row["id"]), score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]


def _build_search_root(tmp_path, snapshot_name):
    snapshot_root = tmp_path / "data" / "benchmark" / snapshot_name
    search_root = tmp_path / "artifacts" / "search" / snapshot_name
    return snapshot_root, search_root


def _make_engine(search_root):
    return SearchEngine(search_root, searcher_factory=lambda _index_dir: _FakeSearcher(search_root / "corpus.jsonl"))


def test_search_respects_source_and_time_filters(tmp_path):
    snapshot_root, search_root = _build_search_root(tmp_path, "s-test")
    _write_jsonl(
        snapshot_root / "info" / "paper" / "arxiv" / "records.jsonl",
        [
            {
                "id": "paper-1",
                "kind": "info",
                "source": "paper/arxiv",
                "timestamp": "2026-01-10",
                "url": "https://arxiv.org/abs/test",
                "payload": {
                    "title": "Inflation Forecasting with Transformers",
                    "description": "Forecast inflation with time-series transformers.",
                    "content": "This paper studies inflation forecasting and macro prediction.",
                },
            }
        ],
    )
    _write_jsonl(
        snapshot_root / "kb" / "report" / "world_bank" / "records.jsonl",
        [
            {
                "id": "report-1",
                "kind": "kb",
                "source": "report/world_bank",
                "timestamp": "2026-02-15",
                "url": "https://worldbank.org/report",
                "payload": {
                    "title": "Climate Adaptation Report",
                    "content": "This report covers climate adaptation and resilience.",
                },
            }
        ],
    )

    build_corpus(snapshot_root, search_root / "corpus.jsonl")
    engine = _make_engine(search_root)

    paper_hits = engine.search("inflation forecast", source="info/paper", time="2026-02-01")
    assert paper_hits["hits"]
    assert paper_hits["hits"][0]["source"] == "info/paper"

    report_hits = engine.search("climate adaptation", source="kb/report", time="2026-02-01")
    assert report_hits["hits"] == []


def test_search_bm25_ranks_more_focused_document_higher(tmp_path):
    snapshot_root, search_root = _build_search_root(tmp_path, "s-bm25")
    filler = " ".join(f"noise{i}" for i in range(200))
    _write_jsonl(
        snapshot_root / "info" / "paper" / "arxiv" / "records.jsonl",
        [
            {
                "id": "focused-doc",
                "kind": "info",
                "source": "paper/arxiv",
                "timestamp": "2026-01-05",
                "url": "https://arxiv.org/abs/focused",
                "payload": {
                    "title": "AAPL price dynamics",
                    "content": "AAPL price AAPL price AAPL price forecast.",
                },
            },
            {
                "id": "diluted-doc",
                "kind": "info",
                "source": "paper/arxiv",
                "timestamp": "2026-01-05",
                "url": "https://arxiv.org/abs/diluted",
                "payload": {
                    "title": "Broad market report",
                    "content": f"AAPL price appears once in a long report {filler}",
                },
            },
        ],
    )

    build_corpus(snapshot_root, search_root / "corpus.jsonl")
    engine = _make_engine(search_root)

    hits = engine.search("AAPL price", source="paper/arxiv", time="2026-02-01", limit=2)

    assert len(hits["hits"]) == 2
    assert hits["hits"][0]["doc_id"] == "focused-doc"


def test_search_supports_blog_and_sociomedia_schema_adaptation(tmp_path):
    snapshot_root = tmp_path / "data" / "mini"
    search_root = tmp_path / "artifacts" / "search" / "mini"
    _write_jsonl(
        snapshot_root / "info" / "blog" / "substack" / "records.jsonl",
        [
            {
                "id": "blog-1",
                "kind": "info",
                "source": "blog/substack",
                "timestamp": "2026-01-10",
                "url": "https://example.com/blog-1",
                "payload": {
                    "title": "Energy transition notes",
                    "author": "Analyst A",
                    "description": "A short note on grid-scale batteries.",
                    "content": "Battery storage deployment is accelerating across utilities.",
                },
            }
        ],
    )
    _write_jsonl(
        snapshot_root / "info" / "sociomedia" / "reddit" / "records.jsonl",
        [
            {
                "id": "reddit-1",
                "kind": "info",
                "source": "sociomedia/reddit",
                "timestamp": "2026-01-09",
                "url": "https://reddit.com/r/test",
                "payload": {
                    "subreddit": "energy",
                    "score": 42,
                    "num_comments": 3,
                    "title": "Battery storage prices keep falling",
                    "comments": [
                        "Utility scale deployments look cheaper this year.",
                        "Interesting signal for storage adoption.",
                    ],
                },
            }
        ],
    )

    build_corpus(snapshot_root, search_root / "corpus.jsonl")
    engine = _make_engine(search_root)

    blog_hits = engine.search("grid battery storage", source="blog", time="2026-01-10")
    reddit_hits = engine.search("storage adoption", source="sociomedia", time="2026-01-10")

    assert blog_hits["hits"][0]["source"] == "info/blog"
    assert blog_hits["hits"][0]["source_type"] == "blog"
    assert blog_hits["hits"][0]["doc_id"] == "blog-1"
    assert "author: Analyst A" in blog_hits["hits"][0]["content"]
    assert reddit_hits["hits"][0]["source_type"] == "sociomedia"
    assert reddit_hits["hits"][0]["doc_id"] == "reddit-1"
    assert "Utility scale deployments look cheaper this year." in reddit_hits["hits"][0]["content"]


def test_search_can_return_multiple_chunks_from_same_document(tmp_path):
    snapshot_root = tmp_path / "data" / "mini"
    search_root = tmp_path / "artifacts" / "search" / "mini"
    long_content = " ".join(["AAPL earnings beat guidance"] * 250)
    _write_jsonl(
        snapshot_root / "info" / "news" / "cnn" / "records.jsonl",
        [
            {
                "id": "news-dup",
                "kind": "info",
                "source": "news/cnn",
                "timestamp": "2026-01-10",
                "url": "https://example.com/news-dup",
                "payload": {
                    "title": "AAPL earnings preview",
                    "description": "AAPL earnings setup",
                    "content": long_content,
                },
            },
            {
                "id": "news-other",
                "kind": "info",
                "source": "news/cnn",
                "timestamp": "2026-01-10",
                "url": "https://example.com/news-other",
                "payload": {
                    "title": "MSFT cloud demand",
                    "description": "MSFT demand remains strong",
                    "content": "MSFT Azure demand looks solid.",
                },
            },
        ],
    )

    build_corpus(snapshot_root, search_root / "corpus.jsonl")
    engine = _make_engine(search_root)

    hits = engine.search("AAPL earnings guidance", source="news", time="2026-01-10", limit=5)

    assert [hit["doc_id"] for hit in hits["hits"]].count("news-dup") >= 2


def test_search_api_returns_content_and_writes_log(tmp_path):
    snapshot_root, search_root = _build_search_root(tmp_path, "s-api")
    _write_jsonl(
        snapshot_root / "info" / "news" / "cnn" / "records.jsonl",
        [
            {
                "id": "news-1",
                "kind": "info",
                "source": "news/cnn",
                "timestamp": "2026-01-10",
                "payload": {"title": "AAPL before cutoff", "content": "AAPL moved lower."},
            }
        ],
    )
    build_corpus(snapshot_root, search_root / "corpus.jsonl")
    log_dir = tmp_path / "logs" / "search"
    client = TestClient(create_app(search_root, log_dir=log_dir, searcher_factory=lambda _index_dir: _FakeSearcher(search_root / "corpus.jsonl")))
    search_client = SearchClient("http://testserver", client=client)

    result = search_client.search("AAPL", time="2026-01-10", source="news", limit=3)

    assert result["hits"]
    assert result["hits"][0]["content"] == "AAPL moved lower."
    assert "id" not in result["hits"][0]
    assert "request_id" not in result
    assert "record_id" not in result["hits"][0]
    request_log = log_dir / "requests.jsonl"
    assert request_log.exists()
    log_row = json.loads(request_log.read_text(encoding="utf-8").splitlines()[0])
    assert log_row["request"]["question"] == "AAPL"
    assert log_row["response"]["hits"][0]["content"] == "AAPL moved lower."


def test_search_builds_and_queries_bm25s_index(tmp_path):
    snapshot_root, search_root = _build_search_root(tmp_path, "s-real")
    _write_jsonl(
        snapshot_root / "info" / "news" / "cnn" / "records.jsonl",
        [
            {
                "id": "news-1",
                "kind": "info",
                "source": "news/cnn",
                "timestamp": "2026-01-10",
                "payload": {
                    "title": "AAPL earnings preview",
                    "description": "AAPL setup",
                    "content": "AAPL earnings beat guidance and margins expanded.",
                },
            },
            {
                "id": "news-2",
                "kind": "info",
                "source": "news/cnn",
                "timestamp": "2026-01-10",
                "payload": {
                    "title": "MSFT cloud demand",
                    "description": "MSFT setup",
                    "content": "MSFT Azure growth remained strong.",
                },
            },
        ],
    )
    build_corpus(snapshot_root, search_root / "corpus.jsonl")
    build_bm25_index(search_root / "corpus.jsonl", search_root / "bm25")

    engine = SearchEngine(search_root)
    result = engine.search("AAPL earnings guidance", source="news", time="2026-01-10", limit=1)

    assert len(result["hits"]) == 1
    assert result["hits"][0]["doc_id"] == "news-1"


def test_search_bm25_handles_duplicate_passage_ids_without_index_error(tmp_path):
    snapshot_root, search_root = _build_search_root(tmp_path, "s-dup-id")
    _write_jsonl(
        snapshot_root / "info" / "news" / "cnn" / "records.jsonl",
        [
            {
                "id": "dup-doc",
                "kind": "info",
                "source": "news/cnn",
                "timestamp": "2026-01-10",
                "payload": {
                    "title": "AAPL demand cools",
                    "content": "AAPL demand weakened after guidance cut.",
                },
            }
        ],
    )
    _write_jsonl(
        snapshot_root / "info" / "news" / "reuters" / "records.jsonl",
        [
            {
                "id": "dup-doc",
                "kind": "info",
                "source": "news/reuters",
                "timestamp": "2026-01-10",
                "payload": {
                    "title": "AAPL suppliers cautious",
                    "content": "Suppliers signaled weaker AAPL demand in Asia.",
                },
            }
        ],
    )
    build_corpus(snapshot_root, search_root / "corpus.jsonl")
    build_bm25_index(search_root / "corpus.jsonl", search_root / "bm25")

    engine = SearchEngine(search_root)
    result = engine.search("AAPL demand", source="news", time="2026-01-10", limit=2)

    assert len(result["hits"]) == 2
    assert {hit["source"] for hit in result["hits"]} == {"info/news"}
