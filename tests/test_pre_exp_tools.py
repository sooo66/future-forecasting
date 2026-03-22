from __future__ import annotations

import json
import re

from agent.tools import OpenBBTool, SearchTool
from forecasting.question_tools import ResidentCodeInterpreterTool
from tools.corpus import build_corpus
from tools.search import SearchEngine


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


def test_search_tool_clamps_to_cutoff_and_truncates_by_tokens(tmp_path):
    snapshot_root = tmp_path / "data" / "mini"
    search_root = tmp_path / "artifacts" / "search" / "mini"
    _write_jsonl(
        snapshot_root / "info" / "news" / "cnn" / "records.jsonl",
        [
            {
                "id": "news-before",
                "kind": "info",
                "source": "news/cnn",
                "timestamp": "2026-01-10",
                "payload": {
                    "title": "AAPL before cutoff",
                    "content": "AAPL moved lower. " * 80,
                },
            },
            {
                "id": "news-after",
                "kind": "info",
                "source": "news/cnn",
                "timestamp": "2026-02-10",
                "payload": {"title": "AAPL after cutoff", "content": "AAPL moved higher."},
            },
            {
                "id": "news-before-2",
                "kind": "info",
                "source": "news/cnn",
                "timestamp": "2026-01-12",
                "payload": {"title": "AAPL still before cutoff", "content": "AAPL stayed weak."},
            },
        ],
    )
    build_corpus(snapshot_root, search_root / "corpus.jsonl")
    tool = SearchTool(
        search_client=SearchEngine(search_root, searcher_factory=lambda _index_dir: _FakeSearcher(search_root / "corpus.jsonl")),
        limit=1,
        content_tokens=12,
    )

    result = tool.call({"question": "before cutoff", "source": "news"}, cuttime="2026-01-20T00:00:00Z")

    assert [hit["doc_id"] for hit in result["hits"]] == ["news-before"]
    assert result["hits"][0]["content"].endswith("...")
    assert result["hits"][0]["content"] != ("AAPL moved lower. " * 80).strip()


def test_search_tool_blocks_repeated_queries_and_formats_model_payload(tmp_path):
    snapshot_root = tmp_path / "data" / "mini"
    search_root = tmp_path / "artifacts" / "search" / "mini"
    _write_jsonl(
        snapshot_root / "info" / "news" / "cnn" / "records.jsonl",
        [
            {
                "id": "news-1",
                "kind": "info",
                "source": "news/cnn",
                "timestamp": "2026-01-10",
                "payload": {"title": "AAPL before cutoff", "content": "AAPL moved lower. " * 40},
            },
        ],
    )
    build_corpus(snapshot_root, search_root / "corpus.jsonl")
    tool = SearchTool(
        search_client=SearchEngine(search_root, searcher_factory=lambda _index_dir: _FakeSearcher(search_root / "corpus.jsonl")),
        limit=3,
        content_tokens=16,
    )

    first = tool.call({"question": "AAPL moved lower", "source": "news"}, cuttime="2026-01-20T00:00:00Z")
    repeated = tool.call({"question": "aapl   moved lower", "source": "news"}, cuttime="2026-01-20T00:00:00Z")
    model_payload = tool.format_result_for_model(first)

    assert repeated["warning"].startswith("Repeated search query blocked")
    assert list(model_payload) == ["count", "hits"]
    assert isinstance(model_payload["hits"][0], str)
    assert "doc_id" not in model_payload["hits"][0]


def test_openbb_tool_clamps_dates_and_maps_quote_alias(monkeypatch):
    captured = {}

    def fake_call(function: str, *, params: dict[str, object], limit: int = 20):
        captured["function"] = function
        captured["params"] = dict(params)
        return {
            "function": function,
            "params": dict(params),
            "result_count": 2,
            "results": [
                {"date": "2026-02-09", "close": 99.0},
                {"date": "2026-02-10", "close": 100.0},
            ],
        }

    monkeypatch.setattr("agent.tools.call_openbb_function", fake_call)
    tool = OpenBBTool()

    result = tool.call(
        {
            "function": "equity.price.quote",
            "params": {"symbol": "AAPL", "end_date": "2026-03-01"},
        },
        cuttime="2026-02-10T12:00:00Z",
    )

    assert captured["function"] == "equity.price.historical"
    assert captured["params"]["end_date"] == "2026-02-10"
    assert len(result["results"]) == 1
    assert result["results"][0]["date"] == "2026-02-10"


def test_resident_code_interpreter_initializes_once_and_resets_between_questions(monkeypatch, tmp_path):
    calls = []

    def fake_call(self, params, files=None, timeout=30, **kwargs):
        calls.append({"params": params, "timeout": timeout})
        return "ok"

    monkeypatch.setattr("forecasting.question_tools.CodeInterpreter.call", fake_call)
    tool = ResidentCodeInterpreterTool(work_dir=tmp_path / "ci")

    tool.begin_question("q1")
    tool.call({"code": "print(1)"})
    tool.begin_question("q2")
    tool.call({"code": "print(2)"})

    assert len(calls) == 4
    assert "_FF_BASELINE_GLOBALS" in json.loads(calls[0]["params"])["code"]
    assert calls[1]["params"] == {"code": "print(1)"}
    assert "_FF_BASELINE_GLOBALS" in json.loads(calls[2]["params"])["code"]
    assert calls[3]["params"] == {"code": "print(2)"}
