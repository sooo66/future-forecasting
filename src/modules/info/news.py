"""News module adapter into unified text schema."""
from __future__ import annotations

import asyncio
from typing import Iterable
from urllib.parse import urlparse

from loguru import logger

from core.contracts import TextRecord, stable_record_id
from modules.base import RunContext
from modules.common.proxy_pool import describe_proxy_mode
from modules.info.news_stack.gdelt.downloader import GDELTDownloader
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
        date_span_days = max(1, (ctx.date_to.date() - ctx.date_from.date()).days + 1)
        long_range_mode = date_span_days > 31
        concurrent_crawls = min(base_cfg.concurrent_crawls, 4) if long_range_mode else base_cfg.concurrent_crawls
        scrapling_primary_concurrency = int(
            base_cfg.get("crawler.scrapling_primary_concurrency", base_cfg.concurrent_crawls)
        )
        scrapling_primary_per_domain_concurrency = int(
            base_cfg.get("crawler.scrapling_primary_per_domain_concurrency", 2)
        )
        scrapling_primary_delay_min_sec = float(
            base_cfg.get("crawler.scrapling_primary_delay_min_sec", 0.0)
        )
        scrapling_primary_delay_max_sec = float(
            base_cfg.get("crawler.scrapling_primary_delay_max_sec", 0.0)
        )
        fallback_concurrency = int(
            base_cfg.get("crawler.fallback_concurrency", base_cfg.concurrent_crawls)
        )
        news_crawl_batch_size = int(base_cfg.get("crawler.news_crawl_batch_size", 250))
        if long_range_mode:
            scrapling_primary_concurrency = min(scrapling_primary_concurrency, 6)
            scrapling_primary_per_domain_concurrency = min(scrapling_primary_per_domain_concurrency, 2)
            fallback_concurrency = min(fallback_concurrency, 4)
            news_crawl_batch_size = min(news_crawl_batch_size, 180)

        whitelist = [
            {"name": name, "domain": domain}
            for domain, name in base_cfg.media_domains.items()
        ]

        cfg = {
            "general": {
                "start_date": ctx.date_from.date().isoformat(),
                "end_date": ctx.date_to.date().isoformat(),
                "concurrent_downloads": base_cfg.concurrent_downloads,
                "concurrent_crawls": concurrent_crawls,
                "request_delay_min": base_cfg.request_delay_min,
                "request_delay_max": base_cfg.request_delay_max,
                "use_proxy": base_cfg.use_proxy,
            },
            "browser": {
                "timeout": base_cfg.browser_timeout,
            },
            "crawler": {
                "primary_engine": base_cfg.get("crawler.primary_engine", "crawl4ai"),
                "scrapling_primary_concurrency": scrapling_primary_concurrency,
                "scrapling_primary_per_domain_concurrency": scrapling_primary_per_domain_concurrency,
                "scrapling_primary_delay_min_sec": scrapling_primary_delay_min_sec,
                "scrapling_primary_delay_max_sec": scrapling_primary_delay_max_sec,
                "scrapling_quality_min_chars": base_cfg.get("crawler.scrapling_quality_min_chars", 280),
                "scrapling_primary_timeout_sec": base_cfg.get("crawler.scrapling_primary_timeout_sec", 12.0),
                "scrapling_primary_retries": base_cfg.get("crawler.scrapling_primary_retries", 1),
                "scrapling_connect_direct_retry": base_cfg.get("crawler.scrapling_connect_direct_retry", True),
                "scrapling_stealthy_headers": base_cfg.get("crawler.scrapling_stealthy_headers", True),
                "scrapling_follow_redirects": base_cfg.get("crawler.scrapling_follow_redirects", True),
                "scrapling_max_redirects": base_cfg.get("crawler.scrapling_max_redirects", 8),
                "scrapling_verify_tls": base_cfg.get("crawler.scrapling_verify_tls", True),
                "scrapling_retry_delay_sec": base_cfg.get("crawler.scrapling_retry_delay_sec", 0.8),
                "scrapling_status_retry_attempts": base_cfg.get("crawler.scrapling_status_retry_attempts", 2),
                "scrapling_status_retry_codes": base_cfg.get(
                    "crawler.scrapling_status_retry_codes",
                    [403, 408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 524],
                ),
                "scrapling_status_retry_backoff_min_sec": base_cfg.get(
                    "crawler.scrapling_status_retry_backoff_min_sec",
                    0.6,
                ),
                "scrapling_status_retry_backoff_max_sec": base_cfg.get(
                    "crawler.scrapling_status_retry_backoff_max_sec",
                    1.8,
                ),
                "scrapling_impersonate_pool": base_cfg.get(
                    "crawler.scrapling_impersonate_pool",
                    ["chrome", "edge", "safari", "firefox"],
                ),
                "scrapling_http3_mode": base_cfg.get("crawler.scrapling_http3_mode", "auto"),
                "scrapling_primary_chunk_size": base_cfg.get("crawler.scrapling_primary_chunk_size", 64),
                "scrapling_short_circuit_sample_size": base_cfg.get(
                    "crawler.scrapling_short_circuit_sample_size",
                    80,
                ),
                "scrapling_short_circuit_fail_ratio": base_cfg.get(
                    "crawler.scrapling_short_circuit_fail_ratio",
                    0.85,
                ),
                "scrapling_domain_connect_failure_streak_threshold": base_cfg.get(
                    "crawler.scrapling_domain_connect_failure_streak_threshold",
                    3,
                ),
                "enable_crawl4ai_rescue_after_scrapling": base_cfg.get(
                    "crawler.enable_crawl4ai_rescue_after_scrapling",
                    True,
                ),
                "enable_scrapling_fallback": base_cfg.get("crawler.enable_scrapling_fallback", True),
                "enable_jina_reader_fallback": base_cfg.get("crawler.enable_jina_reader_fallback", True),
                "fallback_concurrency": fallback_concurrency,
                "fallback_timeout_sec": base_cfg.get("crawler.fallback_timeout_sec", base_cfg.browser_timeout),
                "jina_reader_prefix": base_cfg.get("crawler.jina_reader_prefix", "https://r.jina.ai/"),
                "use_paid_proxy": base_cfg.get("crawler.use_paid_proxy", False),
                "paid_proxy_provider": base_cfg.get("crawler.paid_proxy_provider", ""),
                "paid_proxy_spec": base_cfg.get("crawler.paid_proxy_spec", ""),
                "paid_proxy_host": base_cfg.get("crawler.paid_proxy_host", ""),
                "paid_proxy_port": base_cfg.get("crawler.paid_proxy_port", ""),
                "paid_proxy_username": base_cfg.get("crawler.paid_proxy_username", ""),
                "paid_proxy_password": base_cfg.get("crawler.paid_proxy_password", ""),
                "paid_proxy_session_count": base_cfg.get("crawler.paid_proxy_session_count", 1),
                "paid_proxy_enable_session_suffix": base_cfg.get(
                    "crawler.paid_proxy_enable_session_suffix",
                    False,
                ),
                "proxy_file": base_cfg.get("crawler.proxy_file", ""),
                "proxy_sample_size": base_cfg.get("crawler.proxy_sample_size", 0),
                "proxy_min_quality_score": base_cfg.get("crawler.proxy_min_quality_score", 0.0),
                "enable_network_debug_logs": base_cfg.get("crawler.enable_network_debug_logs", False),
                "news_crawl_batch_size": news_crawl_batch_size,
                "news_reset_in_progress_on_start": base_cfg.get(
                    "crawler.news_reset_in_progress_on_start",
                    True,
                ),
                "news_batch_interleave_domains": base_cfg.get("crawler.news_batch_interleave_domains", True),
                "news_progress_bar_color": base_cfg.get("crawler.news_progress_bar_color", "yellow"),
                "gkg_parse_chunk_size": base_cfg.get("crawler.gkg_parse_chunk_size", 10000),
                "cleanup_raw_gkg_after_build": base_cfg.get(
                    "crawler.cleanup_raw_gkg_after_build",
                    True,
                ),
                "export_url_pool_jsonl": base_cfg.get("crawler.export_url_pool_jsonl", False),
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
        logger.info(
            f"[{self.name}] tuned config: date_span_days="
            f"{(ctx.date_to.date() - ctx.date_from.date()).days + 1} "
            f"concurrent_crawls={config.concurrent_crawls} "
            f"scrapling_primary_concurrency={config.get('crawler.scrapling_primary_concurrency')} "
            f"scrapling_primary_per_domain={config.get('crawler.scrapling_primary_per_domain_concurrency')} "
            f"scrapling_primary_delay="
            f"{config.get('crawler.scrapling_primary_delay_min_sec')}-"
            f"{config.get('crawler.scrapling_primary_delay_max_sec')} "
            f"scrapling_direct_retry={config.get('crawler.scrapling_connect_direct_retry')} "
            f"scrapling_headers={config.get('crawler.scrapling_stealthy_headers')} "
            f"scrapling_http3={config.get('crawler.scrapling_http3_mode')} "
            f"scrapling_status_retry={config.get('crawler.scrapling_status_retry_attempts')} "
            f"use_paid_proxy={config.get('crawler.use_paid_proxy')} "
            f"proxy_sample_size={config.get('crawler.proxy_sample_size')} "
            f"proxy_min_score={config.get('crawler.proxy_min_quality_score')} "
            f"network_debug={config.get('crawler.enable_network_debug_logs')} "
            f"news_crawl_batch_size={config.get('crawler.news_crawl_batch_size')}"
        )
        logger.info(f"[{self.name}] proxy mode={describe_proxy_mode()}")

        builder = URLPoolBuilder(config)
        existing_stats = builder.get_statistics() if ctx.resume else {"total": 0}
        if ctx.resume and int(existing_stats.get("total", 0)) > 0:
            done = int(existing_stats.get("success", 0)) + int(existing_stats.get("failed", 0))
            logger.info(
                f"[{self.name}] resume mode: reuse existing url pool "
                f"done={done}/{existing_stats.get('total', 0)} "
                f"success={existing_stats.get('success', 0)} "
                f"failed={existing_stats.get('failed', 0)} "
                f"pending={existing_stats.get('pending', 0)} "
                f"in_progress={existing_stats.get('in_progress', 0)}"
            )
        else:
            downloader = GDELTDownloader(config)
            downloader.download()
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
                    logger.debug(f"[{self.name}] streaming normalized records={yielded}")
                yield row
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            asyncio.set_event_loop(None)

        logger.info(f"[{self.name}] crawled records={crawled}, yielded={yielded}")
        if yielded == 0:
            logger.warning(f"[{self.name}] no normalized records emitted")

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
