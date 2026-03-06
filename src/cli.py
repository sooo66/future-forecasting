"""Unified CLI for info/kb pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from core.runner import build_run_config, run_modules
from modules import DEFAULT_MODULES
from utils.config import Config
from utils.logger import setup_logger


def _parse_modules(value: str) -> list[str]:
    if not value or value.strip().lower() == "all":
        return ["all"]
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified info/kb pipeline runner")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run modules into one benchmark snapshot")
    run.add_argument("--snapshot", required=True, help="Snapshot id, e.g. s2026_03_static")
    run.add_argument("--from", dest="date_from", required=True, help="Start date (YYYY-MM-DD)")
    run.add_argument("--to", dest="date_to", required=True, help="End date (YYYY-MM-DD)")
    run.add_argument("--kb-from", dest="kb_date_from", default="", help="Optional KB start date (YYYY-MM-DD)")
    run.add_argument("--kb-to", dest="kb_date_to", default="", help="Optional KB end date (YYYY-MM-DD)")
    run.add_argument(
        "--modules",
        default="all",
        help=f"Comma-separated module list or 'all'. default=all ({', '.join(DEFAULT_MODULES)})",
    )
    run.add_argument("--resume", action="store_true", help="Resume from module states")
    run.add_argument("--verbose", action="store_true", help="Enable DEBUG logs on console")
    run.add_argument(
        "--snapshot-base-dir",
        default="data/benchmark",
        help="Base dir for snapshots (default: data/benchmark)",
    )
    run.add_argument(
        "--module-workers",
        type=int,
        default=0,
        help="Parallel workers for modules. 0 means auto (all selected modules).",
    )

    args = parser.parse_args()
    modules = _parse_modules(args.modules)
    cfg = build_run_config(
        snapshot_id=args.snapshot,
        date_from=args.date_from,
        date_to=args.date_to,
        kb_date_from=args.kb_date_from,
        kb_date_to=args.kb_date_to,
        modules=modules,
        resume=bool(args.resume),
        snapshot_base_dir=args.snapshot_base_dir,
        module_workers=args.module_workers,
        project_root=Path(__file__).resolve().parents[1],
    )
    setup_logger(
        Config({"paths": {"log_dir": str(cfg.snapshot_root / "logs")}}),
        verbose=bool(args.verbose),
    )
    result = run_modules(cfg)
    logger.info(f"run completed: {result['stats']}")
    logger.info(f"snapshot root: {result['snapshot_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
