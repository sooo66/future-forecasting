"""Run news pipeline standalone."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

from info.news import GDELTDownloader, GDELTParser, URLPoolBuilder, NewsCrawler
from info.crawler import Crawler
from utils.config import Config
from utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run news pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.toml",
        help="Config path (default: config/settings.toml)",
    )
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    subparsers = parser.add_subparsers(dest="command", help="News commands")

    subparsers.add_parser("download-gkg", help="Download GDELT GKG data")
    subparsers.add_parser("parse-gkg", help="Parse GDELT GKG data")
    subparsers.add_parser("build-url-pool", help="Build URL pool")

    crawl_parser = subparsers.add_parser("crawl", help="Crawl news")
    crawl_parser.add_argument("--limit", type=int, help="Limit URLs")
    crawl_parser.add_argument("--source", type=str, help="urls.jsonl path")
    crawl_parser.add_argument("--output", type=str, help="Output jsonl path")
    crawl_parser.add_argument("--proxy-file", type=str, help="Proxy pool config")
    crawl_parser.add_argument("--no-proxy", action="store_true", help="Disable proxy")
    crawl_parser.add_argument("--verbose", action="store_true", help="Verbose logs")

    pipeline_parser = subparsers.add_parser("full-pipeline", help="Run full pipeline")
    pipeline_parser.add_argument("--limit", type=int, help="Limit URLs")
    pipeline_parser.add_argument("--verbose", action="store_true", help="Verbose logs")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    config = _load_config_with_dates(args.config, args.start_date, args.end_date)
    setup_logger(config, verbose=bool(getattr(args, "verbose", False)))

    if args.command == "download-gkg":
        downloader = GDELTDownloader(config)
        logger.info(f"Download result: {downloader.download()}")
        return

    if args.command == "parse-gkg":
        parser_mod = GDELTParser(config)
        logger.info(f"Parse result: {parser_mod.parse_all()}")
        return

    if args.command == "build-url-pool":
        builder = URLPoolBuilder(config)
        logger.info(f"Build result: {builder.build()}")
        logger.info(f"Stats: {builder.get_statistics()}")
        return

    if args.command == "crawl":
        if args.source:
            output_path = Path(args.output) if args.output else None
            proxy_file = Path(args.proxy_file) if args.proxy_file else None
            use_proxy = False if args.no_proxy else None
            crawler = Crawler(
                config=config,
                source_path=Path(args.source),
                output_path=output_path,
                proxy_file=proxy_file,
                use_proxy=use_proxy,
            )
            records = asyncio.run(crawler.run())
            logger.info(f"Records: {len(records or [])}")
            return

        builder = URLPoolBuilder(config)
        crawler = NewsCrawler(config, builder)
        limit = args.limit if args.limit else None
        records = asyncio.run(crawler.crawl(limit=limit)) or []
        logger.info(f"Records: {len(records)}")
        return

    if args.command == "full-pipeline":
        downloader = GDELTDownloader(config)
        download_result = downloader.download()
        parser_mod = GDELTParser(config)
        parse_result = parser_mod.parse_all()
        builder = URLPoolBuilder(config)
        build_result = builder.build()
        stats = builder.get_statistics()
        crawler = NewsCrawler(config, builder)
        limit = args.limit if args.limit else None
        records = asyncio.run(crawler.crawl(limit=limit)) or []

        logger.info(f"Download: {download_result}")
        logger.info(f"Parse: {parse_result}")
        logger.info(f"Build: {build_result}")
        logger.info(f"URL pool stats: {stats}")
        logger.info(f"Crawl records: {len(records)}")
        return


if __name__ == "__main__":
    main()


def _load_config_with_dates(config_path: str, start_date: str, end_date: str) -> Config:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    raw = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    raw.setdefault("general", {})
    raw["general"]["start_date"] = _normalize_date(start_date, is_end=False)
    raw["general"]["end_date"] = _normalize_date(end_date, is_end=True)
    return Config(raw)


def _normalize_date(value: str, *, is_end: bool) -> str:
    if "T" in value or ":" in value:
        return value
    if is_end:
        return f"{value}T23:59:59+00:00"
    return f"{value}T00:00:00+00:00"
