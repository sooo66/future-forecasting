"""Run crawler standalone."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from loguru import logger

from info.crawler import Crawler
from utils.config import Config
from utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run crawler")
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.toml",
        help="Config path (default: config/settings.toml)",
    )
    parser.add_argument("--source", type=str, help="urls.jsonl path")
    parser.add_argument("--url", action="append", help="Single URL (repeatable)")
    parser.add_argument("--output", type=str, help="Output jsonl path")
    parser.add_argument("--proxy-file", type=str, help="Proxy pool config")
    parser.add_argument("--no-proxy", action="store_true", help="Disable proxy")
    parser.add_argument("--verbose", action="store_true", help="Verbose logs")
    args = parser.parse_args()

    if not args.source and not args.url:
        parser.error("--source or --url is required")

    config = Config(args.config)
    setup_logger(config, verbose=bool(args.verbose))

    output_path = Path(args.output) if args.output else None
    proxy_file = Path(args.proxy_file) if args.proxy_file else None
    use_proxy = False if args.no_proxy else None

    crawler = Crawler(
        config=config,
        source_path=Path(args.source) if args.source else None,
        urls=args.url,
        output_path=output_path,
        proxy_file=proxy_file,
        use_proxy=use_proxy,
    )
    records = asyncio.run(crawler.run())
    logger.info(f"Records: {len(records or [])}")


if __name__ == "__main__":
    main()
