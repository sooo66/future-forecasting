"""URL 池爬虫（兼容旧流程）"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import AsyncIterator, List, Optional

from loguru import logger

from modules.common.crawler import Crawler
from utils.config import Config
from utils.models import Record
from .url_pool.builder import URLPoolBuilder


class NewsCrawler:
    """从 URL 池数据库读取待爬取 URL，复用渐进式爬虫"""

    def __init__(self, config: Config, builder: URLPoolBuilder) -> None:
        self.config = config
        self.builder = builder

    async def crawl(self, limit: Optional[int] = None) -> List[Record]:
        records: List[Record] = []
        async for record in self.crawl_stream(limit=limit):
            records.append(record)
        return records

    async def crawl_stream(self, limit: Optional[int] = None) -> AsyncIterator[Record]:
        urls = self._load_pending_urls(limit)
        if not urls:
            logger.warning("URL 池中没有待爬取 URL")
            return

        crawler = Crawler(
            config=self.config,
            urls=urls,
            output_path=self._default_output_path(),
        )
        async for record in crawler.run_stream():
            yield record

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
