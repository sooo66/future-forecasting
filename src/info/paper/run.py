"""Run ArXiv importer standalone."""
from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from info.paper.arxiv import ArxivConfig, ArxivImporter
from utils.importer_base import BaseImporter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ArXiv importer")
    parser.add_argument("--start-date", required=True, help="ISO8601 start datetime")
    parser.add_argument("--end-date", required=True, help="ISO8601 end datetime")
    parser.add_argument(
        "--output",
        default="data/processed/arxiv/arxiv.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    start_date = _normalize_date(args.start_date, is_end=False)
    end_date = _normalize_date(args.end_date, is_end=True)

    config = ArxivConfig(
        start_date=start_date,
        end_date=end_date,
        output_path=Path(args.output),
        page_size=args.page_size,
        timeout=args.timeout,
    )
    importer = ArxivImporter(config)
    records = importer.run()
    BaseImporter._write_jsonl(records, Path(args.output))
    logger.info(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()


def _normalize_date(value: str, *, is_end: bool) -> str:
    if "T" in value or ":" in value:
        return value
    if is_end:
        return f"{value}T23:59:59+00:00"
    return f"{value}T00:00:00+00:00"
