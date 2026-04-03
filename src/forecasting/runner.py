"""Unified runner for spec-driven forecasting experiments."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

from forecasting.contracts import ForecastResult, MethodArtifact, MethodRuntimeContext
from forecasting.datasets import load_fixed_question_subset
from forecasting.evaluation import render_experiment_summary, summarize_results
from forecasting.llm import OpenAIChatModel
from forecasting.methods._shared import build_failed_result, serialize_config
from forecasting.registry import get_method
from tools.search_clients import build_search_client, resolve_search_backend

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at import time
    tqdm = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)
SUMMARY_FILENAME = "summary.md"


@dataclass
class MethodExecution:
    results: list[ForecastResult]
    artifacts: list[MethodArtifact]


class _NullProgressBar:
    def set_postfix_str(self, _: str, refresh: bool = True) -> None:
        return

    def update(self, _: int = 1) -> None:
        return

    def close(self) -> None:
        return


def run_experiment(
    spec: Any,
    *,
    project_root: Path,
    methods_override: Sequence[str] | None = None,
    output_dir_override: str | None = None,
    search_api_base: str | None = None,
    search_retrieval_mode: str | None = None,
    search_backend: str | None = None,
    force: bool = False,
    max_parallel_methods: int | None = None,
) -> dict[str, Any]:
    output_dir = (project_root / (output_dir_override or spec.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_dataset = load_fixed_question_subset(
        project_root,
        spec.dataset_file,
    )
    subset = list(loaded_dataset["questions"])

    llm = OpenAIChatModel(project_root)
    expected_snapshot_root = str((project_root / spec.knowledge_root).resolve())
    resolved_search_backend = resolve_search_backend(search_backend)
    search_client = build_search_client(
        base_url=search_api_base,
        default_mode=search_retrieval_mode,
        backend=resolved_search_backend,
        exa_base_url=None,
    )
    search_health = search_client.health()
    actual_snapshot_root = str(search_health.get("snapshot_root") or "")
    if resolved_search_backend == "local" and actual_snapshot_root and actual_snapshot_root != expected_snapshot_root:
        raise ValueError(
            f"search service snapshot mismatch: expected {expected_snapshot_root}, got {actual_snapshot_root}"
        )
    runtime_ctx = MethodRuntimeContext(
        project_root=project_root,
        output_dir=output_dir,
        search_engine=search_client,
        llm_factory=lambda: OpenAIChatModel(project_root),
    )
    selected_methods = _resolve_method_ids(spec, methods_override)
    if not selected_methods:
        raise SystemExit("No methods selected after applying filters.")

    summary: dict[str, Any] = {
        "experiment_id": spec.experiment_id,
        "model": llm.model,
        "base_url": llm.base_url,
        "subset_size": len(subset),
        "horizon": loaded_dataset["horizon"],
        "settings": {
            "knowledge_root": spec.knowledge_root,
            "search_backend": resolved_search_backend,
            "search_api_base": search_client.base_url,
            "search_retrieval_mode": getattr(search_client, "default_mode", None)
            or search_health.get("default_mode")
            or "bm25",
            "search_snapshot_root": actual_snapshot_root or expected_snapshot_root,
            "dataset": {
                "dataset_file": spec.dataset_file,
                "subset_id": loaded_dataset["subset_id"],
                "horizon": loaded_dataset["horizon"],
            },
            "method_ids": selected_methods,
            "method_configs": {
                method_id: serialize_config(spec.method_configs.get(method_id))
                for method_id in selected_methods
            },
            "max_parallel_methods": int(max_parallel_methods or spec.max_parallel_methods),
        },
        "question_subset_stats": loaded_dataset["stats"],
        "methods": {},
    }
    summary_path = output_dir / SUMMARY_FILENAME
    _write_summary(summary_path, summary)
    LOGGER.info("initialized experiment summary: %s", summary_path)

    method_runs = _execute_methods(
        selected_methods,
        questions=subset,
        runtime_ctx=runtime_ctx,
        method_configs=spec.method_configs,
        output_dir=output_dir,
        force=force,
        max_parallel_methods=int(max_parallel_methods or spec.max_parallel_methods),
        on_method_complete=lambda method_name, execution: _update_experiment_summary(
            summary,
            summary_path=summary_path,
            method_name=method_name,
            execution=execution,
        ),
    )
    _write_summary(summary_path, summary)
    LOGGER.info("forecasting experiment finished; summary written to %s", summary_path)
    return {"summary": summary, "output_dir": str(output_dir)}


def _resolve_method_ids(spec: Any, methods_override: Sequence[str] | None) -> list[str]:
    method_ids = list(methods_override or spec.method_ids)
    seen: set[str] = set()
    ordered: list[str] = []
    for method_id in method_ids:
        if method_id in seen:
            continue
        seen.add(method_id)
        ordered.append(method_id)
    return ordered


def _execute_methods(
    method_ids: list[str],
    *,
    questions: list[dict[str, Any]],
    runtime_ctx: MethodRuntimeContext,
    method_configs: dict[str, Any],
    output_dir: Path,
    force: bool,
    max_parallel_methods: int,
    on_method_complete: Callable[[str, MethodExecution], None] | None = None,
) -> dict[str, MethodExecution]:
    jobs = [
        {
            "method_id": method_id,
            "position": index,
            "runner": lambda method_id=method_id, progress_position=index: _run_method(
                method_id,
                questions=questions,
                runtime_ctx=runtime_ctx,
                method_config=method_configs.get(method_id),
                output_dir=output_dir,
                force=force,
                progress_position=progress_position,
            ),
        }
        for index, method_id in enumerate(method_ids)
    ]
    if max_parallel_methods <= 1 or len(jobs) <= 1:
        results: dict[str, MethodExecution] = {}
        total_jobs = len(jobs)
        for index, job in enumerate(jobs, start=1):
            LOGGER.info("starting method %s/%s: %s", index, total_jobs, job["method_id"])
            execution = job["runner"]()
            results[job["method_id"]] = execution
            if on_method_complete is not None:
                on_method_complete(job["method_id"], execution)
            LOGGER.info("completed method %s/%s: %s", index, total_jobs, job["method_id"])
        return results

    worker_count = max(1, min(int(max_parallel_methods), len(jobs)))
    LOGGER.info("running %s forecasting methods with max_parallel_methods=%s", len(jobs), worker_count)
    results: dict[str, MethodExecution] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_id = {executor.submit(job["runner"]): job["method_id"] for job in jobs}
        total_jobs = len(future_to_id)
        completed_jobs = 0
        for future in as_completed(future_to_id):
            method_id = future_to_id[future]
            execution = future.result()
            results[method_id] = execution
            completed_jobs += 1
            if on_method_complete is not None:
                on_method_complete(method_id, execution)
            LOGGER.info("completed method %s/%s: %s", completed_jobs, total_jobs, method_id)
    return results


def _run_method(
    method_id: str,
    *,
    questions: list[dict[str, Any]],
    runtime_ctx: MethodRuntimeContext,
    method_config: Any | None,
    output_dir: Path,
    force: bool,
    result_path: Path | None = None,
    persist_artifacts: bool = True,
    progress_position: int = 0,
) -> MethodExecution:
    path = result_path or (output_dir / f"results_{method_id}.jsonl")
    if path.exists() and not force:
        LOGGER.info("loading existing results: %s", path)
        return MethodExecution(results=_read_jsonl(path), artifacts=[])

    LOGGER.info("running method and writing: %s", path)
    method = get_method(method_id)
    session = method.build_session(runtime_ctx, method_config)
    results: list[ForecastResult] = []
    _reset_jsonl_file(path)
    ordered_questions = sorted(
        questions,
        key=lambda item: (item["resolve_time"], item["open_time"], item["market_id"]),
    )
    total = len(ordered_questions)
    progress = _create_progress_bar(method_id, total=total, position=progress_position)
    try:
        for index, question in enumerate(ordered_questions, start=1):
            market_id = question["market_id"]
            LOGGER.info("%s started %s/%s market_id=%s", method_id, index, total, market_id)
            try:
                result = session.run_question(question)
            except Exception as exc:
                LOGGER.exception("%s failed for market_id=%s", method_id, market_id)
                result = build_failed_result(question, str(exc))
            results.append(result)
            _append_jsonl_row(path, result)
            progress.set_postfix_str(market_id, refresh=False)
            progress.update(1)
            LOGGER.info(
                "%s completed %s/%s (%.1f%%) market_id=%s",
                method_id,
                index,
                total,
                (index / total) * 100 if total else 100.0,
                market_id,
            )
    finally:
        progress.close()
    artifacts = session.finalize()
    if persist_artifacts:
        _write_artifacts(output_dir, artifacts)
    return MethodExecution(results=results, artifacts=artifacts)


def _update_experiment_summary(
    summary: dict[str, Any],
    *,
    summary_path: Path,
    method_name: str,
    execution: MethodExecution,
) -> None:
    summary["methods"][method_name] = summarize_results(execution.results)
    _write_summary(summary_path, summary)
    LOGGER.info("updated experiment summary with method=%s", method_name)


def _write_artifacts(output_dir: Path, artifacts: list[MethodArtifact]) -> None:
    for artifact in artifacts:
        path = output_dir / artifact.filename
        if artifact.format == "jsonl":
            _write_jsonl(path, list(artifact.payload))
        else:
            _write_json(path, artifact.payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_experiment_summary(summary), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


def _reset_jsonl_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False))
        fh.write("\n")
        fh.flush()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _create_progress_bar(method_id: str, *, total: int, position: int) -> Any:
    if tqdm is None or not sys.stdout.isatty():
        return _NullProgressBar()
    return tqdm(
        total=total,
        desc=method_id,
        unit="q",
        position=position,
        leave=True,
        dynamic_ncols=True,
        file=sys.stdout,
    )
