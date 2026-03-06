"""URL 池爬虫（兼容旧流程）"""
from __future__ import annotations

import json
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
        raw_bar_color = str(self.config.get("crawler.news_progress_bar_color", "yellow")).strip()
        self.progress_bar_color = raw_bar_color or None
        self._bar_color_prefix = self._ansi_color_prefix(self.progress_bar_color)
        self._bar_color_suffix = "\033[0m" if self._bar_color_prefix else ""
        self._progress_desc = self._colorize_text("爬取 news URLs", self.progress_bar_color)

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
        output_records, recovered_success = self._recover_success_status_from_output(output_path)
        if recovered_success > 0:
            logger.info(
                f"news resume 状态恢复: marked_success={recovered_success} "
                f"from_existing_output_records={output_records}"
            )
        initial_stats = self.builder.get_statistics()
        initial_success = int(initial_stats.get("success", 0))
        initial_failed = int(initial_stats.get("failed", 0))
        total_success = initial_success
        total_failed = initial_failed
        total_urls = int(initial_stats.get("total", 0))
        pending_total = int(initial_stats.get("pending", 0))
        initial_completed = initial_success + initial_failed
        output_crawled_hint = min(output_records, total_urls) if total_urls > 0 else output_records
        display_total = total_urls if total_urls > 0 else output_crawled_hint
        display_completed = max(initial_completed, output_crawled_hint)
        display_completed = min(display_completed, display_total) if display_total > 0 else display_completed
        initial_remaining = max(0, display_total - display_completed)
        logger.info(
            f"news URL 池进度基线: crawled={display_completed} "
            f"remaining={initial_remaining} total={display_total} "
            f"(status_success={initial_success} status_failed={initial_failed} "
            f"pending={pending_total} output_records={output_records})"
        )
        if remaining is None:
            progress_total = display_total
            progress_initial = display_completed
        else:
            progress_total = min(display_total, display_completed + remaining)
            progress_initial = min(display_completed, progress_total)

        with tqdm(
            total=progress_total,
            initial=progress_initial,
            desc=self._progress_desc,
            unit="url",
            dynamic_ncols=True,
            mininterval=0.5,
            file=sys.stdout,
            colour=self.progress_bar_color,
            bar_format=self._build_progress_bar_format(),
        ) as pbar:
            display_crawled = progress_initial

            def _refresh_postfix() -> None:
                nonlocal display_crawled
                status_crawled = total_success + total_failed
                if display_total > 0:
                    status_crawled = min(status_crawled, display_total)
                display_crawled = max(display_crawled, status_crawled)
                pbar.set_postfix(
                    crawled=f"{display_crawled}/{display_total}",
                    remaining=max(0, display_total - display_crawled),
                    total=display_total,
                    success=total_success,
                    failed=total_failed,
                )

            _refresh_postfix()
            while remaining is None or remaining > 0:
                batch_limit = self.batch_size if remaining is None else min(self.batch_size, remaining)
                urls = self.builder.reserve_pending_urls(batch_limit)
                if not urls:
                    if batch_idx == 0:
                        if initial_remaining == 0 and display_total > 0:
                            logger.info(
                                f"URL 池已无剩余任务: crawled={display_completed} "
                                f"remaining=0 total={display_total}"
                            )
                        else:
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
                progress_room = max(0, progress_total - pbar.n)
                if progress_room > 0:
                    pbar.update(min(len(urls), progress_room))
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
                _refresh_postfix()
                logger.debug(
                    f"URL 批次完成: batch={batch_idx} reserved={len(urls)} "
                    f"success={batch_success} failed={batch_failed}"
                )

                if remaining is not None:
                    remaining -= len(urls)

        final_status_crawled = total_success + total_failed
        if display_total > 0:
            final_status_crawled = min(final_status_crawled, display_total)
        final_crawled = max(display_completed, final_status_crawled)
        final_remaining = max(0, display_total - final_crawled)
        logger.info(
            f"URL 池爬取完成: batches={batch_idx} reserved={total_reserved} "
            f"success={total_success} failed={total_failed} "
            f"crawled={final_crawled} remaining={final_remaining} total={display_total}"
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

    def _build_progress_bar_format(self) -> str:
        if self._bar_color_prefix:
            return f"{{l_bar}}{self._bar_color_prefix}{{bar}}{self._bar_color_suffix}{{r_bar}}"
        return "{l_bar}{bar}{r_bar}"

    @staticmethod
    def _ansi_color_prefix(color_name: Optional[str]) -> str:
        if not color_name:
            return ""
        name = str(color_name).strip().lower()
        mapping = {
            "black": "\033[30m",
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "white": "\033[37m",
        }
        return mapping.get(name, "")

    @classmethod
    def _colorize_text(cls, text: str, color_name: Optional[str]) -> str:
        prefix = cls._ansi_color_prefix(color_name)
        if not prefix:
            return text
        return f"{prefix}{text}\033[0m"

    def _recover_success_status_from_output(self, output_path: Path, *, chunk_size: int = 400) -> tuple[int, int]:
        if not output_path.exists():
            return 0, 0

        found_records = 0
        updated_rows = 0
        buffered_urls: list[str] = []

        def _flush_buffer() -> None:
            nonlocal updated_rows
            if not buffered_urls:
                return
            deduped_urls = list(dict.fromkeys(buffered_urls))
            updated_rows += self.builder.bulk_update_status_by_url(deduped_urls, status="success")
            buffered_urls.clear()

        try:
            with output_path.open("r", encoding="utf-8") as fp:
                for raw_line in fp:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    url = str(payload.get("url") or "").strip()
                    if not url:
                        continue
                    found_records += 1
                    buffered_urls.append(url)
                    if len(buffered_urls) >= chunk_size:
                        _flush_buffer()
            _flush_buffer()
        except OSError as exc:
            logger.warning(f"读取历史爬取输出失败，跳过状态恢复: {exc}")
            return 0, 0

        return found_records, updated_rows

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
