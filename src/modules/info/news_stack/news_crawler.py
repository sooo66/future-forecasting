"""URL 池爬虫（兼容旧流程）"""
from __future__ import annotations

import sys
import sqlite3
from collections import Counter
from pathlib import Path
from typing import AsyncIterator, List, Optional

from loguru import logger
from tqdm import tqdm

from modules.common.crawler import Crawler
from utils.config import Config
from utils.models import Record
from .url_pool.builder import URLPoolBuilder


class NewsCrawler:
    """从 URL 池数据库读取待爬取 URL，复用渐进式爬虫"""

    def __init__(self, config: Config, builder: URLPoolBuilder) -> None:
        self.config = config
        self.builder = builder
        self.batch_size = max(1, int(self.config.get("crawler.news_crawl_batch_size", 250)))
        self.reset_in_progress_on_start = bool(
            self.config.get("crawler.news_reset_in_progress_on_start", True)
        )

    async def crawl(self, limit: Optional[int] = None) -> List[Record]:
        records: List[Record] = []
        async for record in self.crawl_stream(limit=limit):
            records.append(record)
        return records

    async def crawl_stream(self, limit: Optional[int] = None) -> AsyncIterator[Record]:
        if self.reset_in_progress_on_start:
            restored = self.builder.reset_status(
                from_status="in_progress",
                to_status="pending",
                error="requeued_after_interruption",
            )
            if restored:
                logger.warning(f"已重置中断批次 URL: {restored}")

        remaining = max(0, int(limit)) if limit is not None else None
        batch_idx = 0
        total_reserved = 0
        output_path = self._default_output_path()
        initial_stats = self.builder.get_statistics()
        initial_success = int(initial_stats.get("success", 0))
        initial_failed = int(initial_stats.get("failed", 0))
        total_success = initial_success
        total_failed = initial_failed
        total_urls = int(initial_stats.get("total", 0))
        pending_total = int(initial_stats.get("pending", 0))
        initial_completed = initial_success + initial_failed
        logger.info(
            f"news URL 池进度基线: done={initial_completed}/{total_urls} "
            f"success={initial_success} failed={initial_failed} pending={pending_total}"
        )
        if remaining is None:
            progress_total = total_urls
            progress_initial = initial_completed
        else:
            progress_total = min(total_urls, initial_completed + remaining)
            progress_initial = min(initial_completed, progress_total)

        with tqdm(
            total=progress_total,
            initial=progress_initial,
            desc="爬取 news URLs",
            unit="url",
            dynamic_ncols=True,
            mininterval=0.5,
            file=sys.stdout,
        ) as pbar:
            pbar.set_postfix(
                success=total_success,
                failed=total_failed,
                pending=max(0, total_urls - (total_success + total_failed)),
            )
            while remaining is None or remaining > 0:
                batch_limit = self.batch_size if remaining is None else min(self.batch_size, remaining)
                urls = self.builder.reserve_pending_urls(batch_limit)
                if not urls:
                    if batch_idx == 0:
                        logger.warning("URL 池中没有待爬取 URL")
                    break

                batch_idx += 1
                total_reserved += len(urls)
                reserved_urls = {str(item.get("url") or "").strip() for item in urls if item.get("url")}
                logger.debug(
                    f"开始爬取 URL 批次: batch={batch_idx} size={len(urls)} "
                    f"remaining_limit={remaining if remaining is not None else 'all'}"
                )

                crawler = Crawler(
                    config=self.config,
                    urls=urls,
                    output_path=output_path,
                )

                try:
                    async for record in crawler.run_stream():
                        yield record
                except Exception as exc:
                    succeeded_urls = crawler.last_succeeded_urls
                    unresolved_urls = reserved_urls - succeeded_urls
                    self.builder.bulk_update_status_by_url(list(succeeded_urls), status="success")
                    self.builder.bulk_update_status_by_url(
                        list(unresolved_urls),
                        status="pending",
                        error=f"batch_interrupted:{type(exc).__name__}",
                    )
                    logger.exception(f"URL 批次爬取异常，已将未完成 URL 放回 pending: batch={batch_idx}")
                    raise

                succeeded_urls = crawler.last_succeeded_urls
                failed_urls = crawler.last_failed_urls
                failed_reasons = crawler.last_failed_reasons
                self.builder.bulk_update_status_by_url(list(succeeded_urls), status="success")
                self.builder.bulk_update_status_by_url(list(failed_urls), status="failed", error="crawl_failed")
                pbar.update(len(urls))
                batch_failed = len(failed_urls)
                batch_success = len(succeeded_urls)
                total_success += batch_success
                total_failed += batch_failed
                if failed_urls:
                    self._apply_failure_reasons(failed_reasons)
                    logger.warning(
                        f"URL 批次失败摘要: batch={batch_idx} "
                        f"failed={batch_failed} top_reasons={self._summarize_failure_reasons(failed_reasons)}"
                    )
                pbar.set_postfix(
                    success=total_success,
                    failed=total_failed,
                    pending=max(0, total_urls - (total_success + total_failed)),
                )
                logger.debug(
                    f"URL 批次完成: batch={batch_idx} reserved={len(urls)} "
                    f"success={batch_success} failed={batch_failed}"
                )

                if remaining is not None:
                    remaining -= len(urls)

        logger.info(
            f"URL 池爬取完成: batches={batch_idx} reserved={total_reserved} "
            f"success={total_success} failed={total_failed}"
        )

    def _load_pending_urls(self, limit: Optional[int]) -> List[dict]:
        db_path = Path(self.builder.url_pool_db)
        if not db_path.exists():
            logger.error(f"URL 池数据库不存在: {db_path}")
            return []

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        sql = "SELECT url, gkg_date, themes, created_at FROM url_pool WHERE status = 'pending'"
        params = ()
        if limit:
            sql += " LIMIT ?"
            params = (limit,)
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        urls = []
        for row in rows:
            if not row or not row[0]:
                continue
            url, gkg_date, themes, created_at = row
            urls.append(
                {
                    "url": url,
                    # Keep GDELT date as first priority for downstream timestamp.
                    "created_at": gkg_date or created_at,
                    "themes": themes,
                }
            )
        return urls

    def _default_output_path(self) -> Path:
        return self.config.processed_data_dir / "news_crawl_results.jsonl"

    def _apply_failure_reasons(self, failed_reasons: dict[str, str]) -> None:
        grouped: dict[str, list[str]] = {}
        for url, reason in failed_reasons.items():
            grouped.setdefault(reason or "crawl_failed", []).append(url)
        for reason, urls in grouped.items():
            self.builder.bulk_update_status_by_url(urls, status="failed", error=reason)

    @staticmethod
    def _summarize_failure_reasons(failed_reasons: dict[str, str], *, top_k: int = 3) -> str:
        if not failed_reasons:
            return "none"
        counts = Counter(reason or "unknown_failure" for reason in failed_reasons.values())
        parts = [f"{reason} x{count}" for reason, count in counts.most_common(top_k)]
        return "; ".join(parts)
