"""Run Substack importer standalone."""
from __future__ import annotations

import argparse
from dataclasses import replace

from loguru import logger

from info.blog.substack.substack import load_import_config, SubstackImporter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Substack importer")
    parser.add_argument(
        "--config",
        type=str,
        default="config/substack_authors.toml",
        help="Config path (default: config/substack_authors.toml)",
    )
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    start_date = _normalize_date(args.start_date, is_end=False)
    end_date = _normalize_date(args.end_date, is_end=True)
    config = load_import_config(args.config)
    authors = [
        replace(author, start_date=start_date, end_date=end_date)
        for author in config.authors
    ]
    config = replace(config, authors=authors)
    importer = SubstackImporter(config)
    results = importer.run()
    for author, count in results.items():
        logger.info(f"{author}: {count}")


if __name__ == "__main__":
    main()


def _normalize_date(value: str, *, is_end: bool) -> str:
    if "T" in value or ":" in value:
        return value
    if is_end:
        return f"{value}T23:59:59+00:00"
    return f"{value}T00:00:00+00:00"
