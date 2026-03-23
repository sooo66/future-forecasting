"""Offline-indexed search API and client."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import importlib
from itertools import repeat
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Optional, Sequence
from uuid import uuid4

import httpx
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from core.io import MODULE_OUTPUT_PATHS
from tools.dense_embeddings import DEFAULT_DENSE_EMBEDDING_MODEL, build_text_embedder
from tools.rerankers import build_text_reranker
from utils.time_utils import to_day

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional at import time
    tqdm = None  # type: ignore[assignment]


_DEFAULT_BASE_DIR = "data/benchmark"
_DEFAULT_SEARCH_API_BASE = "http://127.0.0.1:8000"
_DEFAULT_LOG_DIR = "logs/search"
_DEFAULT_SEARCH_ARTIFACT_DIR = "artifacts/search"
_DEFAULT_FETCH_MULTIPLIER = 10
_DEFAULT_FETCH_MIN = 50
_MAX_FETCH_LIMIT = 400
_BM25S_METHOD = "lucene"
_RETRIEVAL_MODES = {"bm25", "dense", "hybrid"}
_DEFAULT_RETRIEVAL_MODE = "bm25"
_DEFAULT_RRF_K = 60
_DEFAULT_DENSE_BATCH_SIZE = 64
_DEFAULT_DENSE_AUTO_WORKERS = 0
_DEFAULT_DENSE_MAX_AUTO_WORKERS = 4
_DEFAULT_DENSE_MIN_PARALLEL_ROWS = 256
_DEFAULT_RERANK_CANDIDATE_LIMIT = 40
_DEFAULT_RERANKER_BATCH_SIZE = 32


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
    mode: str | None = None


def resolve_search_api_base(base_url: str | None = None) -> str:
    value = str(base_url or os.getenv("SEARCH_API_BASE") or _DEFAULT_SEARCH_API_BASE).strip()
    if not value:
        return _DEFAULT_SEARCH_API_BASE
    return value.rstrip("/")


def resolve_search_retrieval_mode(mode: str | None = None) -> str | None:
    value = str(mode or os.getenv("SEARCH_RETRIEVAL_MODE") or "").strip().lower()
    if not value:
        return None
    if value not in _RETRIEVAL_MODES:
        raise ValueError(f"Unsupported search retrieval mode: {value}")
    return value


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


def build_dense_index(
    corpus_path: Path,
    index_dir: Path,
    *,
    model_name: str = DEFAULT_DENSE_EMBEDDING_MODEL,
    device: str | None = None,
    batch_size: int = _DEFAULT_DENSE_BATCH_SIZE,
    workers: int = _DEFAULT_DENSE_AUTO_WORKERS,
    overwrite: bool = False,
    show_progress: bool | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    corpus_path = Path(corpus_path).expanduser().resolve()
    index_dir = Path(index_dir).expanduser().resolve()
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file does not exist: {corpus_path}")
    np = _load_numpy_module()
    resolved_batch_size = max(1, int(batch_size or _DEFAULT_DENSE_BATCH_SIZE))
    resolved_device = _resolve_dense_device(device)

    if index_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Index directory already exists: {index_dir}")
        import shutil

        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    rows = list(_iter_corpus_rows(corpus_path))
    texts = [str(row.get("contents") or row.get("content") or row.get("title") or "") for row in rows]
    resolved_workers = _resolve_dense_worker_count(
        requested_workers=workers,
        device=resolved_device,
        row_count=len(texts),
    )
    logger.info(
        "building dense index: passages={} model={} device={} batch_size={} workers={}",
        len(texts),
        model_name,
        resolved_device,
        resolved_batch_size,
        resolved_workers,
    )

    embeddings_path = index_dir / "embeddings.npy"
    matrix = None
    dimension = 0
    completed = 0
    started = time.perf_counter()
    progress_bar = _create_dense_progress_bar(
        total=len(texts),
        show_progress=show_progress,
        description="dense index",
    )

    batch_arrays = _iter_dense_embedding_batches(
        texts,
        model_name=model_name,
        device=resolved_device,
        batch_size=resolved_batch_size,
        workers=resolved_workers,
    )
    try:
        for batch_array in batch_arrays:
            batch_matrix = np.asarray(batch_array, dtype="float32")
            if batch_matrix.ndim != 2:
                raise ValueError("Dense embedding model must return a 2D array-like result")
            if matrix is None:
                dimension = int(batch_matrix.shape[1]) if batch_matrix.shape[0] else 0
                matrix = np.lib.format.open_memmap(
                    embeddings_path,
                    mode="w+",
                    dtype="float32",
                    shape=(len(rows), dimension),
                )
            next_completed = completed + int(batch_matrix.shape[0])
            matrix[completed:next_completed] = batch_matrix
            completed = next_completed
            _update_dense_progress(
                progress_bar,
                batch_size=int(batch_matrix.shape[0]),
                completed=completed,
                total=len(texts),
                started=started,
                callback=progress_callback,
            )
    finally:
        close = getattr(batch_arrays, "close", None)
        if callable(close):
            close()
        if progress_bar is not None:
            progress_bar.close()

    if matrix is None:
        empty = np.empty((0, 0), dtype="float32")
        np.save(embeddings_path, empty, allow_pickle=False)
    else:
        matrix.flush()

    metadata = {
        "model_name": model_name,
        "corpus_size": len(rows),
        "dimension": dimension,
        "device": resolved_device,
        "batch_size": resolved_batch_size,
        "workers": resolved_workers,
    }
    (index_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - started
    logger.info(
        "dense index complete: passages={} dimension={} elapsed={:.2f}s output={}",
        len(rows),
        dimension,
        elapsed,
        index_dir,
    )
    return index_dir


class SearchClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        client: Any | None = None,
        default_mode: str | None = None,
    ) -> None:
        self.base_url = resolve_search_api_base(base_url)
        self.default_mode = resolve_search_retrieval_mode(default_mode)
        self._client = client or httpx.Client(base_url=self.base_url, timeout=30.0, trust_env=False)

    def search(
        self,
        question: str,
        *,
        time: str | None = None,
        source: str | Sequence[str] | None = None,
        limit: int = 5,
        mode: str | None = None,
    ) -> dict[str, Any]:
        source_value: str | None
        if source is None:
            source_value = None
        elif isinstance(source, str):
            source_value = source
        else:
            source_value = ",".join(str(item).strip() for item in source if str(item).strip()) or None
        resolved_mode = resolve_search_retrieval_mode(mode) or self.default_mode
        response = self._client.post(
            "/search",
            json={
                "question": question,
                "time": time,
                "source": source_value,
                "limit": int(limit or 5),
                "mode": resolved_mode,
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
        retrieval_mode: str | None = None,
        reranker: Any | None = None,
        rerank_candidate_limit: int = _DEFAULT_RERANK_CANDIDATE_LIMIT,
    ) -> None:
        self.search_root = Path(search_root).expanduser().resolve()
        self.corpus_path = self.search_root / "corpus.jsonl"
        self.index_dir = self.search_root / "bm25"
        self.dense_index_dir = self.search_root / "dense"
        self.stats_path = self.search_root / "stats.json"
        self._reranker = reranker
        self._rerank_candidate_limit = max(1, int(rerank_candidate_limit or _DEFAULT_RERANK_CANDIDATE_LIMIT))
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Search corpus does not exist: {self.corpus_path}")

        self.stats = _read_json(self.stats_path) if self.stats_path.exists() else {}
        self.snapshot_root = Path(self.stats.get("snapshot_root") or self.search_root).resolve()
        self._rows = _iter_corpus_rows(self.corpus_path)
        self._rows_by_id = _load_rows_by_id(self.corpus_path)
        self._searchers: dict[str, Any] = {}
        if searcher_factory is None:
            if self.index_dir.exists():
                self._searchers["bm25"] = _load_bm25s_searcher(self.index_dir, len(self._rows))
        else:
            self._searchers["bm25"] = searcher_factory(self.index_dir)
        if self.dense_index_dir.exists():
            self._searchers["dense"] = _load_dense_searcher(self.dense_index_dir)
        self.available_modes = self._resolve_available_modes()
        if not self.available_modes:
            raise FileNotFoundError(
                f"No search index found under {self.search_root}. Build BM25 and/or dense indexes first."
            )
        requested_mode = resolve_search_retrieval_mode(retrieval_mode)
        self.retrieval_mode = requested_mode or _select_default_retrieval_mode(self.available_modes)
        if self.retrieval_mode not in self.available_modes:
            raise ValueError(
                f"Retrieval mode {self.retrieval_mode!r} is unavailable. "
                f"Available modes: {', '.join(self.available_modes)}"
            )

    def search(
        self,
        question: str,
        *,
        time: str | None = None,
        source: str | Sequence[str] | None = None,
        limit: int = 5,
        mode: str | None = None,
    ) -> dict[str, Any]:
        query = (question or "").strip()
        if not query:
            raise ValueError("search.question is required")
        retrieval_mode = resolve_search_retrieval_mode(mode) or self.retrieval_mode
        if retrieval_mode not in self.available_modes:
            raise ValueError(
                f"Retrieval mode {retrieval_mode!r} is unavailable. "
                f"Available modes: {', '.join(self.available_modes)}"
            )

        max_hits = max(1, min(int(limit or 5), 20))
        rerank_enabled = self._should_rerank(retrieval_mode)
        target_hits = max_hits if not rerank_enabled else max(max_hits, self._rerank_candidate_limit)
        before_day = _normalize_day(time) if time else None
        source_filters = _parse_source_filters(source)
        filtered_hits: list[dict[str, Any]] = []
        fetch_k = min(_MAX_FETCH_LIMIT, max(_DEFAULT_FETCH_MIN, target_hits * _DEFAULT_FETCH_MULTIPLIER))

        while True:
            raw_hits = list(self._search(query, fetch_k, mode=retrieval_mode))
            filtered_hits = []
            for hit in raw_hits:
                row = _resolve_hit_row(hit, rows=self._rows, rows_by_id=self._rows_by_id)
                if row is None:
                    continue
                if not _row_matches(row, source_filters=source_filters, before_day=before_day):
                    continue
                filtered_hits.append(_row_to_hit(row, score=float(getattr(hit, "score", 0.0))))
                if len(filtered_hits) >= target_hits:
                    break
            if len(filtered_hits) >= target_hits or len(raw_hits) < fetch_k or fetch_k >= _MAX_FETCH_LIMIT:
                break
            fetch_k = min(_MAX_FETCH_LIMIT, fetch_k * 2)

        if rerank_enabled and filtered_hits:
            filtered_hits = self._rerank_hits(query, filtered_hits[:target_hits])

        return {
            "snapshot_root": str(self.snapshot_root),
            "question": query,
            "time": before_day,
            "source": _source_filter_label(source),
            "mode": retrieval_mode,
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
            "dense_index_dir": str(self.dense_index_dir),
            "default_mode": self.retrieval_mode,
            "available_modes": list(self.available_modes),
            "reranker_enabled": self._reranker is not None,
            "rerank_candidate_limit": self._rerank_candidate_limit if self._reranker is not None else 0,
            "reranker_model": getattr(self._reranker, "model_name", None),
            "reranker_device": _reranker_device_label(self._reranker),
        }

    def _resolve_available_modes(self) -> list[str]:
        modes: list[str] = []
        if "bm25" in self._searchers:
            modes.append("bm25")
        if "dense" in self._searchers:
            modes.append("dense")
        if "bm25" in self._searchers and "dense" in self._searchers:
            modes.append("hybrid")
        return modes

    def _search(self, query: str, k: int, *, mode: str) -> list[Any]:
        if mode == "hybrid":
            return self._hybrid_search(query, k)
        searcher = self._searchers.get(mode)
        if searcher is None:
            raise ValueError(f"Searcher for mode {mode!r} is not available")
        return list(searcher.search(query, k))

    def _hybrid_search(self, query: str, k: int) -> list[_SearchHit]:
        bm25_hits = list(self._searchers["bm25"].search(query, k))
        dense_hits = list(self._searchers["dense"].search(query, k))
        fused: dict[int, float] = {}
        for hits in (bm25_hits, dense_hits):
            for rank, hit in enumerate(hits, start=1):
                row_index = int(getattr(hit, "row_index"))
                fused[row_index] = fused.get(row_index, 0.0) + 1.0 / (_DEFAULT_RRF_K + rank)
        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        return [_SearchHit(row_index=row_index, score=score) for row_index, score in ordered[:k]]

    def _should_rerank(self, mode: str) -> bool:
        return self._reranker is not None and mode == "hybrid"

    def _rerank_hits(self, query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        texts = [_hit_text_for_reranking(hit) for hit in hits]
        scores = list(self._reranker.score(query, texts))
        if len(scores) != len(hits):
            raise ValueError(
                f"Reranker returned {len(scores)} scores for {len(hits)} hits"
            )
        ranked: list[tuple[float, float, int, dict[str, Any]]] = []
        for order, (hit, score) in enumerate(zip(hits, scores)):
            retrieval_score = float(hit.get("score", 0.0))
            rerank_score = float(score)
            enriched = dict(hit)
            enriched["retrieval_score"] = round(retrieval_score, 4)
            enriched["rerank_score"] = round(rerank_score, 4)
            enriched["score"] = round(rerank_score, 4)
            ranked.append((rerank_score, retrieval_score, -order, enriched))
        ranked.sort(reverse=True)
        return [item[3] for item in ranked]


def create_app(
    search_root: Path,
    *,
    log_dir: Path | None = None,
    searcher_factory: Callable[[Path], Any] | None = None,
    retrieval_mode: str | None = None,
    reranker: Any | None = None,
    reranker_model_name: str | None = None,
    reranker_device: str | None = None,
    reranker_batch_size: int = _DEFAULT_RERANKER_BATCH_SIZE,
    rerank_candidate_limit: int = _DEFAULT_RERANK_CANDIDATE_LIMIT,
) -> FastAPI:
    resolved_reranker = reranker
    resolved_reranker_model = str(reranker_model_name or "").strip()
    if resolved_reranker is None and resolved_reranker_model:
        logger.info(
            "enabling hybrid reranker: model={} device={} batch_size={} candidates={}",
            resolved_reranker_model,
            (reranker_device or "").strip() or "auto",
            max(1, int(reranker_batch_size or _DEFAULT_RERANKER_BATCH_SIZE)),
            max(1, int(rerank_candidate_limit or _DEFAULT_RERANK_CANDIDATE_LIMIT)),
        )
        resolved_reranker = build_text_reranker(
            model_name=resolved_reranker_model,
            device=(reranker_device or "").strip() or None,
            batch_size=int(reranker_batch_size or _DEFAULT_RERANKER_BATCH_SIZE),
        )
    engine = SearchEngine(
        search_root,
        searcher_factory=searcher_factory,
        retrieval_mode=retrieval_mode,
        reranker=resolved_reranker,
        rerank_candidate_limit=rerank_candidate_limit,
    )
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
            mode=request.mode,
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


def _load_numpy_module() -> Any:
    try:
        return importlib.import_module("numpy")
    except Exception as exc:
        raise RuntimeError("numpy is required to build and serve dense search. Install numpy first.") from exc


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


class _DenseSearcher:
    def __init__(self, *, embeddings: Any, embedder: Any) -> None:
        self._embeddings = embeddings
        self._embedder = embedder
        self._np = _load_numpy_module()

    def search(self, query: str, k: int) -> list[_SearchHit]:
        if len(self._embeddings) == 0:
            return []
        query_vector = self._np.asarray(
            self._embedder.embed_texts([query])[0],
            dtype="float32",
        )
        scores = self._embeddings @ query_vector
        effective_k = min(max(1, int(k)), len(scores))
        if effective_k <= 0:
            return []
        indices = self._np.argpartition(-scores, effective_k - 1)[:effective_k]
        ordered = sorted(
            ((int(index), float(scores[index])) for index in indices),
            key=lambda item: item[1],
            reverse=True,
        )
        return [_SearchHit(row_index=row_index, score=score) for row_index, score in ordered]


def _load_dense_searcher(index_dir: Path) -> Any:
    index_dir = Path(index_dir).expanduser().resolve()
    metadata_path = index_dir / "metadata.json"
    embeddings_path = index_dir / "embeddings.npy"
    if not metadata_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(f"Dense search index is incomplete: {index_dir}")
    metadata = _read_json(metadata_path)
    model_name = str(metadata.get("model_name") or DEFAULT_DENSE_EMBEDDING_MODEL)
    embedder = build_text_embedder(model_name=model_name)
    np = _load_numpy_module()
    embeddings = np.load(embeddings_path, mmap_mode="r")
    return _DenseSearcher(embeddings=embeddings, embedder=embedder)


def _reranker_device_label(reranker: Any | None) -> str | None:
    if reranker is None:
        return None
    value = getattr(reranker, "resolved_device", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raw = getattr(reranker, "device", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _select_default_retrieval_mode(available_modes: Sequence[str]) -> str:
    for candidate in ("hybrid", "dense", "bm25"):
        if candidate in available_modes:
            return candidate
    return available_modes[0]


def _resolve_dense_device(device: str | None) -> str:
    value = str(device or "").strip().lower()
    if value:
        return value
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dense_worker_count(*, requested_workers: int, device: str, row_count: int) -> int:
    requested = int(requested_workers or 0)
    if requested < 0:
        raise ValueError("workers must be >= 0")
    normalized_device = str(device or "cpu").strip().lower()
    if normalized_device and normalized_device != "cpu":
        return 1
    cpu_count = os.cpu_count() or 1
    if requested > 0:
        return max(1, min(requested, cpu_count))
    if cpu_count <= 1 or row_count < _DEFAULT_DENSE_MIN_PARALLEL_ROWS:
        return 1
    return max(1, min(cpu_count, _DEFAULT_DENSE_MAX_AUTO_WORKERS))


def _iter_dense_embedding_batches(
    texts: Sequence[str],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    workers: int,
):
    if not texts:
        return
    if workers <= 1:
        embedder = build_text_embedder(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )
        for batch in _iter_text_batches(texts, batch_size):
            yield embedder.embed_texts(batch)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for batch_embeddings in executor.map(
            _embed_dense_batch_worker,
            _iter_text_batches(texts, batch_size),
            repeat(model_name),
            repeat(device),
            repeat(batch_size),
        ):
            yield batch_embeddings


def _embed_dense_batch_worker(
    texts: Sequence[str],
    model_name: str,
    device: str,
    batch_size: int,
) -> list[list[float]]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if str(device or "").strip().lower() == "cpu":
        try:
            import torch

            torch.set_num_threads(1)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(1)
        except Exception:
            pass
    embedder = build_text_embedder(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    return embedder.embed_texts(list(texts))


def _create_dense_progress_bar(*, total: int, show_progress: bool | None, description: str):
    if total <= 0 or tqdm is None:
        return None
    enabled = bool(show_progress) if show_progress is not None else sys.stderr.isatty()
    if not enabled:
        return None
    return tqdm(
        total=total,
        desc=description,
        unit="passage",
        dynamic_ncols=True,
    )


def _iter_text_batches(texts: Sequence[str], batch_size: int):
    for start in range(0, len(texts), batch_size):
        yield list(texts[start : start + batch_size])


def _update_dense_progress(
    progress_bar: Any,
    *,
    batch_size: int,
    completed: int,
    total: int,
    started: float,
    callback: Callable[[dict[str, Any]], None] | None,
) -> None:
    elapsed = max(0.0, time.perf_counter() - started)
    eta_seconds = None
    if completed > 0 and total > completed:
        rate = completed / max(elapsed, 1e-9)
        if rate > 0:
            eta_seconds = (total - completed) / rate
    if progress_bar is not None:
        progress_bar.update(batch_size)
        if eta_seconds is not None:
            progress_bar.set_postfix_str(f"eta={int(eta_seconds)}s")
    if callback is not None:
        callback(
            {
                "completed": completed,
                "total": total,
                "batch_size": batch_size,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta_seconds,
            }
        )


def _hit_text_for_reranking(hit: dict[str, Any]) -> str:
    title = str(hit.get("title") or "").strip()
    content = str(hit.get("content") or "").strip()
    if not title:
        return content
    if not content:
        return title
    if content.lower().startswith(title.lower()):
        return content
    return f"{title}\n{content}"


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
