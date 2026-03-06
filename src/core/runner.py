"""Unified runner for info/kb modules."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List

from loguru import logger

from core.io import SnapshotPaths, append_jsonl
from core.state import ResumeState
from modules import DEFAULT_MODULES, MODULE_REGISTRY
from modules.base import RunContext
from utils.time_utils import parse_datetime


def _parse_day(value: str, *, end_of_day: bool) -> datetime:
    raw = value.strip()
    if "T" not in raw:
        suffix = "T23:59:59+00:00" if end_of_day else "T00:00:00+00:00"
        raw = raw + suffix
    dt = parse_datetime(raw)
    if dt is None:
        raise ValueError(f"Invalid date: {value}")
    return dt.astimezone(timezone.utc)


@dataclass
class RunConfig:
    snapshot_id: str
    info_date_from: datetime
    info_date_to: datetime
    kb_date_from: datetime
    kb_date_to: datetime
    modules: List[str]
    resume: bool
    module_workers: int
    snapshot_root: Path
    project_root: Path


def build_run_config(
    *,
    snapshot_id: str,
    date_from: str,
    date_to: str,
    kb_date_from: str = "",
    kb_date_to: str = "",
    modules: List[str],
    resume: bool,
    module_workers: int = 0,
    snapshot_base_dir: str = "data/benchmark",
    project_root: Path,
) -> RunConfig:
    if not modules or modules == ["all"]:
        modules = list(DEFAULT_MODULES)

    unknown = [m for m in modules if m not in MODULE_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown modules: {unknown}")

    info_dt_from = _parse_day(date_from, end_of_day=False)
    info_dt_to = _parse_day(date_to, end_of_day=True)
    if info_dt_from > info_dt_to:
        raise ValueError("--from must be <= --to")
    kb_raw_from = (kb_date_from or "").strip()
    kb_raw_to = (kb_date_to or "").strip()
    kb_dt_from = _parse_day(kb_raw_from, end_of_day=False) if kb_raw_from else info_dt_from
    kb_dt_to = _parse_day(kb_raw_to, end_of_day=True) if kb_raw_to else info_dt_to
    if kb_dt_from > kb_dt_to:
        raise ValueError("--kb-from must be <= --kb-to")

    snapshot_root = Path(snapshot_base_dir) / snapshot_id
    return RunConfig(
        snapshot_id=snapshot_id,
        info_date_from=info_dt_from,
        info_date_to=info_dt_to,
        kb_date_from=kb_dt_from,
        kb_date_to=kb_dt_to,
        modules=modules,
        resume=resume,
        module_workers=max(0, int(module_workers)),
        snapshot_root=snapshot_root,
        project_root=project_root,
    )


def _resolve_module_workers(cfg: RunConfig) -> int:
    module_count = len(cfg.modules)
    if module_count <= 1:
        return 1
    if cfg.module_workers <= 0:
        info_span_days = max(1, (cfg.info_date_to.date() - cfg.info_date_from.date()).days + 1)
        if "info.news" in cfg.modules and info_span_days > 31:
            return 1
        return min(module_count, 2)
    return max(1, min(cfg.module_workers, module_count))


def _ordered_modules(cfg: RunConfig) -> List[str]:
    modules = list(cfg.modules)
    info_span_days = max(1, (cfg.info_date_to.date() - cfg.info_date_from.date()).days + 1)
    if "info.news" in modules and info_span_days > 31:
        return [name for name in modules if name != "info.news"] + ["info.news"]
    return modules


def _run_single_module(cfg: RunConfig, paths: SnapshotPaths, name: str) -> dict[str, Any]:
    cls = MODULE_REGISTRY[name]
    module = cls()  # type: ignore[call-arg]
    records_path = paths.records_path(name)
    if not cfg.resume:
        # Non-resume mode should rebuild module outputs deterministically.
        if records_path.exists():
            records_path.unlink()
        state_path = paths.state_path(name)
        if state_path.exists():
            state_path.unlink()
        work_dir = paths.module_work_dir(name)
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    use_kb_range = name.startswith("kb.")
    date_from = cfg.kb_date_from if use_kb_range else cfg.info_date_from
    date_to = cfg.kb_date_to if use_kb_range else cfg.info_date_to

    state = ResumeState(paths.state_path(name), enable_resume=cfg.resume)
    ctx = RunContext(
        snapshot_id=cfg.snapshot_id,
        date_from=date_from,
        date_to=date_to,
        resume=cfg.resume,
        snapshot_paths=paths,
        project_root=cfg.project_root,
    )

    logger.info(f"[runner] start module={name}")
    written = 0
    skipped = 0
    errors = 0
    try:
        rows = module.run(ctx)
        for row in rows:
            rid = str(row.get("id") or "")
            if not rid:
                errors += 1
                continue
            if state.has(rid):
                skipped += 1
                continue
            append_jsonl(records_path, row)
            state.add(rid)
            written += 1
    except Exception as exc:
        logger.exception(f"[runner] module failed: {name} ({exc})")
        errors += 1

    state.save(extra={"module": name, "written": written, "skipped": skipped, "errors": errors})
    logger.info(f"[runner] done module={name} written={written} skipped={skipped} errors={errors}")
    return {
        "records_path": str(records_path),
        "written": written,
        "skipped_by_resume": skipped,
        "errors": errors,
    }


def run_modules(cfg: RunConfig) -> Dict[str, Any]:
    paths = SnapshotPaths.create(cfg.snapshot_root)
    manifest: Dict[str, Any] = {
        "snapshot_id": cfg.snapshot_id,
        "date_from": cfg.info_date_from.isoformat().replace("+00:00", "Z"),
        "date_to": cfg.info_date_to.isoformat().replace("+00:00", "Z"),
        "info_date_from": cfg.info_date_from.isoformat().replace("+00:00", "Z"),
        "info_date_to": cfg.info_date_to.isoformat().replace("+00:00", "Z"),
        "kb_date_from": cfg.kb_date_from.isoformat().replace("+00:00", "Z"),
        "kb_date_to": cfg.kb_date_to.isoformat().replace("+00:00", "Z"),
        "resume": cfg.resume,
        "modules": cfg.modules,
        "results": {},
    }

    workers = _resolve_module_workers(cfg)
    ordered_modules = _ordered_modules(cfg)
    logger.info(f"[runner] module parallelism workers={workers}")
    if ordered_modules != cfg.modules:
        logger.info(f"[runner] module launch order={ordered_modules}")

    if workers == 1:
        for name in ordered_modules:
            manifest["results"][name] = _run_single_module(cfg, paths, name)
    else:
        results: Dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_single_module, cfg, paths, name): name for name in ordered_modules}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as exc:
                    logger.exception(f"[runner] module worker crashed: {name} ({exc})")
                    records_path = paths.records_path(name)
                    results[name] = {
                        "records_path": str(records_path),
                        "written": 0,
                        "skipped_by_resume": 0,
                        "errors": 1,
                    }
        for name in cfg.modules:
            manifest["results"][name] = results.get(
                name,
                {
                    "records_path": str(paths.records_path(name)),
                    "written": 0,
                    "skipped_by_resume": 0,
                    "errors": 1,
                },
            )

    paths.run_manifest_path().write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    totals = {"written": 0, "skipped_by_resume": 0, "errors": 0}
    for v in manifest["results"].values():
        totals["written"] += int(v["written"])
        totals["skipped_by_resume"] += int(v["skipped_by_resume"])
        totals["errors"] += int(v["errors"])

    stats = {
        "snapshot_id": cfg.snapshot_id,
        "module_count": len(cfg.modules),
        "totals": totals,
    }
    paths.stats_path().write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": manifest, "stats": stats, "snapshot_root": str(cfg.snapshot_root)}
