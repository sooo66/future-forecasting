"""Offline-indexed search API and client."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Optional, Sequence
from uuid import uuid4

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from core.io import MODULE_OUTPUT_PATHS
from utils.time_utils import to_day


_DEFAULT_BASE_DIR = "data/benchmark"
_DEFAULT_SEARCH_API_BASE = "http://127.0.0.1:8000"
_DEFAULT_LOG_DIR = "logs/search"
_DEFAULT_SEARCH_ARTIFACT_DIR = "artifacts/search"
_DEFAULT_FETCH_MULTIPLIER = 10
_DEFAULT_FETCH_MIN = 50
_MAX_FETCH_LIMIT = 400
_BM25S_METHOD = "lucene"


@dataclass(frozen=True)
class SourceFilter:
    kind: Optional[str] = None
    source_type: Optional[str] = None
    exact: Optional[str] = None

    def matches(self, source: str) -> bool:
        kind, source_type = _split_coarse_source(source)
        if self.kind and kind != self.kind:
            return False
        if self.source_type and source_type != self.source_type:
            return False
        if self.exact and source != self.exact:
            return False
        return True


class SearchRequest(BaseModel):
    question: str
    time: str | None = None
    source: str | None = None
    limit: int = 5


def resolve_search_api_base(base_url: str | None = None) -> str:
    value = str(base_url or os.getenv("SEARCH_API_BASE") or _DEFAULT_SEARCH_API_BASE).strip()
    if not value:
        return _DEFAULT_SEARCH_API_BASE
    return value.rstrip("/")


def default_search_log_dir(project_root: Path) -> Path:
    return (project_root / _DEFAULT_LOG_DIR).resolve()


def default_search_root(project_root: Path, snapshot_root: Path) -> Path:
    return (project_root / _DEFAULT_SEARCH_ARTIFACT_DIR / Path(snapshot_root).resolve().name).resolve()


def find_latest_snapshot_root(project_root: Path, *, base_dir: str = _DEFAULT_BASE_DIR) -> Path:
    benchmark_root = (project_root / base_dir).resolve()
    if not benchmark_root.exists():
        raise FileNotFoundError(f"Benchmark directory does not exist: {benchmark_root}")

    candidates = []
    for entry in benchmark_root.iterdir():
        if not entry.is_dir():
            continue
        if any((entry / rel).exists() for rel in MODULE_OUTPUT_PATHS.values()):
            candidates.append(entry)

    if not candidates:
        raise FileNotFoundError(f"No snapshot with records.jsonl found under: {benchmark_root}")

    return max(candidates, key=lambda item: item.stat().st_mtime)


def build_bm25_index(corpus_path: Path, index_dir: Path, *, overwrite: bool = False) -> Path:
    corpus_path = Path(corpus_path).expanduser().resolve()
    index_dir = Path(index_dir).expanduser().resolve()
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file does not exist: {corpus_path}")
    bm25s = _load_bm25s_module()

    if index_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Index directory already exists: {index_dir}")
        import shutil

        shutil.rmtree(index_dir)
    index_dir.parent.mkdir(parents=True, exist_ok=True)

    rows = list(_iter_corpus_rows(corpus_path))
    texts = [str(row.get("contents") or "") for row in rows]
    tokenized_corpus = bm25s.tokenize(texts, show_progress=False)
    retriever = bm25s.BM25(method=_BM25S_METHOD)
    retriever.index(tokenized_corpus, show_progress=False)
    retriever.save(index_dir)
    return index_dir


class SearchClient:
    def __init__(self, base_url: str | None = None, *, client: Any | None = None) -> None:
        self.base_url = resolve_search_api_base(base_url)
        self._client = client or httpx.Client(base_url=self.base_url, timeout=30.0, trust_env=False)

    def search(
        self,
        question: str,
        *,
        time: str | None = None,
        source: str | Sequence[str] | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        source_value: str | None
        if source is None:
            source_value = None
        elif isinstance(source, str):
            source_value = source
        else:
            source_value = ",".join(str(item).strip() for item in source if str(item).strip()) or None
        response = self._client.post(
            "/search",
            json={
                "question": question,
                "time": time,
                "source": source_value,
                "limit": int(limit or 5),
            },
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> dict[str, Any]:
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()


class SearchEngine:
    def __init__(
        self,
        search_root: Path,
        *,
        searcher_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        self.search_root = Path(search_root).expanduser().resolve()
        self.corpus_path = self.search_root / "corpus.jsonl"
        self.index_dir = self.search_root / "bm25"
        self.stats_path = self.search_root / "stats.json"
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Search corpus does not exist: {self.corpus_path}")

        self.stats = _read_json(self.stats_path) if self.stats_path.exists() else {}
        self.snapshot_root = Path(self.stats.get("snapshot_root") or self.search_root).resolve()
        self._rows = _iter_corpus_rows(self.corpus_path)
        self._rows_by_id = _load_rows_by_id(self.corpus_path)
        if searcher_factory is None:
            self._searcher = _load_bm25s_searcher(self.index_dir, len(self._rows))
        else:
            self._searcher = searcher_factory(self.index_dir)

    def search(
        self,
        question: str,
        *,
        time: str | None = None,
        source: str | Sequence[str] | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        query = (question or "").strip()
        if not query:
            raise ValueError("search.question is required")

        max_hits = max(1, min(int(limit or 5), 20))
        before_day = _normalize_day(time) if time else None
        source_filters = _parse_source_filters(source)
        filtered_hits: list[dict[str, Any]] = []
        fetch_k = min(_MAX_FETCH_LIMIT, max(_DEFAULT_FETCH_MIN, max_hits * _DEFAULT_FETCH_MULTIPLIER))

        while True:
            raw_hits = list(self._searcher.search(query, fetch_k))
            filtered_hits = []
            for hit in raw_hits:
                row = _resolve_hit_row(hit, rows=self._rows, rows_by_id=self._rows_by_id)
                if row is None:
                    continue
                if not _row_matches(row, source_filters=source_filters, before_day=before_day):
                    continue
                filtered_hits.append(_row_to_hit(row, score=float(getattr(hit, "score", 0.0))))
                if len(filtered_hits) >= max_hits:
                    break
            if len(filtered_hits) >= max_hits or len(raw_hits) < fetch_k or fetch_k >= _MAX_FETCH_LIMIT:
                break
            fetch_k = min(_MAX_FETCH_LIMIT, fetch_k * 2)

        return {
            "snapshot_root": str(self.snapshot_root),
            "question": query,
            "time": before_day,
            "source": _source_filter_label(source),
            "total_candidates": len(filtered_hits),
            "hits": filtered_hits[:max_hits],
        }

    def health(self) -> dict[str, Any]:
        return {
            "snapshot_root": str(self.snapshot_root),
            "record_count": len(self._rows),
            "indexed_documents": len(self._rows),
            "search_root": str(self.search_root),
            "corpus_path": str(self.corpus_path),
            "index_dir": str(self.index_dir),
        }


def create_app(
    search_root: Path,
    *,
    log_dir: Path | None = None,
    searcher_factory: Callable[[Path], Any] | None = None,
) -> FastAPI:
    engine = SearchEngine(search_root, searcher_factory=searcher_factory)
    request_log_path = ((log_dir or Path.cwd() / _DEFAULT_LOG_DIR).resolve() / "requests.jsonl")
    app = FastAPI(title="future-forecasting-search", version="2.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return engine.health()

    @app.post("/search")
    def search(request: SearchRequest) -> dict[str, Any]:
        request_id = str(uuid4())
        started = time.perf_counter()
        response = engine.search(
            request.question,
            time=request.time,
            source=request.source,
            limit=request.limit,
        )
        elapsed = time.perf_counter() - started
        _write_jsonl_row(
            request_log_path,
            {
                "request_id": request_id,
                "request": request.model_dump(),
                "response": response,
                "latency_ms": round(elapsed * 1000, 2),
                "logged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        return response

    return app


def _load_bm25s_module() -> Any:
    try:
        return importlib.import_module("bm25s")
    except Exception as exc:
        raise RuntimeError("bm25s is required to build and serve BM25 search. Install bm25s first.") from exc


@dataclass(frozen=True)
class _SearchHit:
    row_index: int
    score: float


class _Bm25sSearcher:
    def __init__(self, retriever: Any, corpus_size: int) -> None:
        self._retriever = retriever
        self._row_indices = list(range(corpus_size))
        self._bm25s = _load_bm25s_module()

    def search(self, query: str, k: int) -> list[_SearchHit]:
        effective_k = min(max(1, int(k)), len(self._row_indices))
        if effective_k <= 0:
            return []
        query_tokens = self._bm25s.tokenize([query], show_progress=False)
        results = self._retriever.retrieve(
            query_tokens,
            corpus=self._row_indices,
            k=effective_k,
            show_progress=False,
            return_as="tuple",
        )
        row_indices = results.documents[0].tolist() if hasattr(results.documents[0], "tolist") else list(results.documents[0])
        scores = results.scores[0].tolist() if hasattr(results.scores[0], "tolist") else list(results.scores[0])
        return [_SearchHit(row_index=int(row_index), score=float(score)) for row_index, score in zip(row_indices, scores)]


def _load_bm25s_searcher(index_dir: Path, corpus_size: int) -> Any:
    index_dir = Path(index_dir).expanduser().resolve()
    if not index_dir.exists():
        raise FileNotFoundError(f"Search index does not exist: {index_dir}")
    bm25s = _load_bm25s_module()
    retriever = bm25s.BM25.load(index_dir, load_corpus=False, mmap=True)
    return _Bm25sSearcher(retriever, corpus_size)


def _iter_corpus_rows(corpus_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_rows_by_id(corpus_path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _iter_corpus_rows(corpus_path):
        row_id = str(row.get("id") or "").strip()
        if row_id:
            rows[row_id] = row
    return rows


def _resolve_hit_row(
    hit: Any,
    *,
    rows: list[dict[str, Any]],
    rows_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    row_index = getattr(hit, "row_index", None)
    if isinstance(row_index, int) and 0 <= row_index < len(rows):
        return rows[row_index]
    docid = str(getattr(hit, "docid", "")).strip()
    if not docid:
        return None
    return rows_by_id.get(docid)


def _row_matches(
    row: dict[str, Any],
    *,
    source_filters: list[SourceFilter],
    before_day: str | None,
) -> bool:
    if before_day and str(row.get("timestamp") or "") > before_day:
        return False
    if not source_filters:
        return True
    source = str(row.get("source") or "")
    return any(item.matches(source) for item in source_filters)


def _row_to_hit(row: dict[str, Any], *, score: float) -> dict[str, Any]:
    source = str(row.get("source") or "")
    _kind, source_type = _split_coarse_source(source)
    return {
        "doc_id": str(row.get("doc_id") or ""),
        "score": round(float(score), 4),
        "source": source,
        "source_type": source_type,
        "timestamp": str(row.get("timestamp") or ""),
        "title": str(row.get("title") or ""),
        "content": str(row.get("content") or ""),
        "url": row.get("url"),
    }


def _split_coarse_source(source: str) -> tuple[str, str]:
    if "/" not in source:
        lowered = source.strip().lower()
        return lowered, lowered
    kind, source_type = source.split("/", 1)
    return kind.strip().lower(), source_type.strip().lower()


def _write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False))
        fh.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_day(value: Any) -> str | None:
    day = to_day(value)
    if day:
        return day
    text = str(value or "").strip()
    if not text:
        return None
    raise ValueError(f"Invalid date/time value: {value}")


def _parse_source_filters(source: Optional[str | Sequence[str]]) -> list[SourceFilter]:
    if source is None:
        return []
    if isinstance(source, str):
        raw_items = [item.strip().lower() for item in source.split(",") if item.strip()]
    else:
        raw_items = [str(item).strip().lower() for item in source if str(item).strip()]
    filters: list[SourceFilter] = []
    for raw in raw_items:
        if not raw or raw == "all":
            continue
        if raw in {"info", "kb"}:
            filters.append(SourceFilter(kind=raw))
            continue
        if raw in {"news", "paper", "blog", "sociomedia", "book", "report"}:
            filters.append(SourceFilter(source_type=raw))
            continue
        if raw.startswith("info/") or raw.startswith("kb/"):
            kind, source_type = _split_coarse_source(raw)
            filters.append(SourceFilter(kind=kind, source_type=source_type, exact=f"{kind}/{source_type}"))
            continue
        if "/" in raw:
            first, _rest = raw.split("/", 1)
            if first in {"news", "paper", "blog", "sociomedia", "book", "report"}:
                filters.append(SourceFilter(source_type=first))
                continue
        filters.append(SourceFilter(source_type=raw))
    return filters


def _source_filter_label(source: Optional[str | Sequence[str]]) -> str:
    if source is None:
        return "all"
    if isinstance(source, str):
        return source or "all"
    values = [str(item).strip() for item in source if str(item).strip()]
    return ",".join(values) or "all"
