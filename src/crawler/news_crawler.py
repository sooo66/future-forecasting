"""URL 池爬虫（兼容旧流程）"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional

from loguru import logger

from .crawler import Crawler
from utils.config import Config
from url_pool_builder.builder import URLPoolBuilder


class NewsCrawler:
    """从 URL 池数据库读取待爬取 URL，复用渐进式爬虫"""

    def __init__(self, config: Config, builder: URLPoolBuilder) -> None:
        self.config = config
        self.builder = builder

    async def crawl(self, limit: Optional[int] = None) -> List[dict]:
        urls = self._load_pending_urls(limit)
        if not urls:
            logger.warning("URL 池中没有待爬取 URL")
            return []

        crawler = Crawler(
            config=self.config,
            urls=urls,
            output_path=self._default_output_path(),
        )
        return await crawler.run()

    def _load_pending_urls(self, limit: Optional[int]) -> List[str]:
        db_path = Path(self.builder.url_pool_db)
        if not db_path.exists():
            logger.error(f"URL 池数据库不存在: {db_path}")
            return []

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        sql = "SELECT url FROM url_pool WHERE status = 'pending'"
        params = ()
        if limit:
            sql += " LIMIT ?"
            params = (limit,)
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        return [row[0] for row in rows if row and row[0]]

    def _default_output_path(self) -> Path:
        return self.config.processed_data_dir / "crawl_results_pool.jsonl"
