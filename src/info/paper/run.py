"""Run ArXiv importer standalone."""
from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from info.paper.arxiv import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_PAGE_SIZE,
    DEFAULT_QUERY_PROFILE,
    DEFAULT_TIMEOUT,
    ArxivConfig,
    ArxivImporter,
    _normalize_date,
)
from utils.importer_base import BaseImporter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ArXiv importer")
    parser.add_argument("--from", dest="from_date", required=True, help="YYYY-MM-DD or ISO8601")
    parser.add_argument("--to", dest="to_date", required=True, help="YYYY-MM-DD or ISO8601")
    parser.add_argument("--start-date", dest="start_date", help="Deprecated alias of --from")
    parser.add_argument("--end-date", dest="end_date", help="Deprecated alias of --to")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path",
    )
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=3.0,
        help="Seconds between page requests to avoid arXiv rate limit (default: 3.0)",
    )
    parser.add_argument(
        "--max-429-retries",
        type=int,
        default=6,
        help="Max retries when arXiv returns HTTP 429 (default: 6)",
    )
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="Use proxy settings from environment variables",
    )
    parser.add_argument(
        "--query-profile",
        choices=["forecasting", "all"],
        default=DEFAULT_QUERY_PROFILE,
        help="Preset query profile: forecasting (default) or all",
    )
    parser.add_argument(
        "--query",
        dest="custom_query",
        help="Custom arXiv search query (without submittedDate clause)",
    )
    args = parser.parse_args()

    from_raw = args.start_date or args.from_date
    to_raw = args.end_date or args.to_date
    from_date = _normalize_date(from_raw, is_end=False)
    to_date = _normalize_date(to_raw, is_end=True)

    config = ArxivConfig(
        from_date=from_date,
        to_date=to_date,
        output_path=Path(args.output),
        page_size=args.page_size,
        timeout=args.timeout,
        query_profile=args.query_profile,
        custom_query=args.custom_query,
        request_interval=args.request_interval,
        max_429_retries=args.max_429_retries,
        use_env_proxy=args.use_env_proxy,
    )
    importer = ArxivImporter(config)
    records = importer.run()
    BaseImporter._write_jsonl(records, Path(args.output))
    logger.info(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
