"""News module adapter into unified text schema."""
from __future__ import annotations

import asyncio
from typing import Iterable
from urllib.parse import urlparse

from loguru import logger

from core.contracts import TextRecord, stable_record_id
from modules.base import RunContext
from modules.info.news_stack.gdelt.downloader import GDELTDownloader
from modules.info.news_stack.gdelt.parser import GDELTParser
from modules.info.news_stack.url_pool.builder import URLPoolBuilder
from modules.info.news_stack.news_crawler import NewsCrawler
from utils.config import Config
from utils.models import Record
from utils.time_utils import to_day


class NewsModule:
    name = "info.news"

    @staticmethod
    def _normalize_news_source(source_value: str, url: str | None) -> str:
        raw = (source_value or "").strip().lower()
        domain = ""
        if raw.startswith("news/"):
            domain = raw.split("/", 1)[1]
        if not domain and url:
            try:
                domain = (urlparse(url).netloc or "").lower()
            except Exception:
                domain = ""
        domain = domain.split("@")[-1].split(":")[0].strip()
        if domain.startswith("www."):
            domain = domain[4:]
        return f"news/{domain or 'unknown'}"

    def _build_config(self, ctx: RunContext) -> Config:
        base_cfg = Config("config/settings.toml")
        work = ctx.snapshot_paths.module_work_dir(self.name)
        raw_dir = work / "raw" / "gkg"
        processed_dir = work / "processed" / "records"
        url_pool_db = work / "url_pool" / "url_pool.db"
        whitelist = [
            {"name": name, "domain": domain}
            for domain, name in base_cfg.media_domains.items()
        ]

        cfg = {
            "general": {
                "start_date": ctx.date_from.date().isoformat(),
                "end_date": ctx.date_to.date().isoformat(),
                "concurrent_downloads": base_cfg.concurrent_downloads,
                "concurrent_crawls": base_cfg.concurrent_crawls,
                "request_delay_min": base_cfg.request_delay_min,
                "request_delay_max": base_cfg.request_delay_max,
                "use_proxy": base_cfg.use_proxy,
            },
            "browser": {
                "timeout": base_cfg.browser_timeout,
            },
            "crawler": {
                "primary_engine": base_cfg.get("crawler.primary_engine", "crawl4ai"),
                "scrapling_primary_concurrency": base_cfg.get(
                    "crawler.scrapling_primary_concurrency",
                    base_cfg.concurrent_crawls,
                ),
                "scrapling_quality_min_chars": base_cfg.get("crawler.scrapling_quality_min_chars", 280),
                "scrapling_primary_timeout_sec": base_cfg.get("crawler.scrapling_primary_timeout_sec", 12.0),
                "scrapling_primary_retries": base_cfg.get("crawler.scrapling_primary_retries", 1),
                "enable_crawl4ai_rescue_after_scrapling": base_cfg.get(
                    "crawler.enable_crawl4ai_rescue_after_scrapling",
                    True,
                ),
                "enable_scrapling_fallback": base_cfg.get("crawler.enable_scrapling_fallback", True),
                "enable_jina_reader_fallback": base_cfg.get("crawler.enable_jina_reader_fallback", True),
                "fallback_concurrency": base_cfg.get("crawler.fallback_concurrency", base_cfg.concurrent_crawls),
                "fallback_timeout_sec": base_cfg.get("crawler.fallback_timeout_sec", base_cfg.browser_timeout),
                "jina_reader_prefix": base_cfg.get("crawler.jina_reader_prefix", "https://r.jina.ai/"),
            },
            "paths": {
                "raw_data_dir": str(raw_dir),
                "processed_data_dir": str(processed_dir),
                "url_pool_db": str(url_pool_db),
                "log_dir": str(work / "logs"),
            },
            "whitelist": {
                "media_domains": whitelist,
            },
        }
        return Config(cfg)

    def run(self, ctx: RunContext) -> Iterable[dict]:
        config = self._build_config(ctx)
        logger.info(f"[{self.name}] running news pipeline from={ctx.date_from.date()} to={ctx.date_to.date()}")

        downloader = GDELTDownloader(config)
        downloader.download()

        parser = GDELTParser(config)
        parser.parse_all()

        builder = URLPoolBuilder(config)
        builder.build()

        crawler = NewsCrawler(config, builder)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        crawled = 0
        yielded = 0
        stream = crawler.crawl_stream()
        try:
            while True:
                try:
                    rec = loop.run_until_complete(stream.__anext__())
                except StopAsyncIteration:
                    break
                crawled += 1
                row = self._to_text_record(rec)
                if row is None:
                    continue
                yielded += 1
                if yielded % 100 == 0:
                    logger.info(f"[{self.name}] streaming normalized records={yielded}")
                yield row
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            asyncio.set_event_loop(None)

        logger.info(f"[{self.name}] crawled records={crawled}, yielded={yielded}")

    def _to_text_record(self, rec: Record) -> dict | None:
        payload = {
            "title": rec.title or "",
            "description": rec.description,
            "content": rec.content or "",
        }
        # News timestamp must prioritize GDELT day injected by URL pool (gkg_date).
        day = to_day(rec.published_at) or to_day(rec.pubtime)
        if not day:
            return None
        source = self._normalize_news_source(rec.source, rec.url)
        rid = stable_record_id(source, rec.url, day, payload.get("title"))
        row = TextRecord(
            id=rid,
            kind="info",
            source=source,
            timestamp=day,
            url=rec.url or None,
            payload=payload,
        ).normalized()
        return row.to_dict()
