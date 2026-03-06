"""URL 池构建模块"""
import json
import sys
import sqlite3
import uuid
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Optional

import pandas as pd
import tldextract
from loguru import logger
from tqdm import tqdm

from utils.config import Config


class URLPoolBuilder:
    """URL 池构建器
    
    从解析后的 GKG 数据中提取 URL，过滤白名单域名，去重并构建 URL 池。
    URL 池存储在 SQLite 数据库中，便于后续查询和更新状态。
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.raw_data_dir = config.raw_data_dir
        self.processed_data_dir = config.processed_data_dir
        self.url_pool_db = config.url_pool_db
        self.whitelist_domains = config.whitelist_domains
        self.media_domains = config.media_domains
        self.gkg_parse_chunk_size = max(1000, int(self.config.get("crawler.gkg_parse_chunk_size", 10000)))
        self.export_url_pool_jsonl = bool(self.config.get("crawler.export_url_pool_jsonl", False))
        self.cleanup_raw_gkg_after_build = bool(self.config.get("crawler.cleanup_raw_gkg_after_build", True))
        self._tldextract = tldextract.TLDExtract(suffix_list_urls=None, cache_dir=False)

        # 确保目录存在
        self.url_pool_db.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self._init_database()
    
    def _init_database(self):
        """初始化 URL 池数据库"""
        conn = sqlite3.connect(self.url_pool_db)
        cursor = conn.cursor()
        
        # 创建 URL 池表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS url_pool (
                id TEXT PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                domain TEXT NOT NULL,
                source_name TEXT,
                gkg_date TEXT,
                gkg_record_id TEXT,
                themes TEXT,
                status TEXT DEFAULT 'pending',
                last_error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_domain ON url_pool(domain)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON url_pool(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_url ON url_pool(url)")
        
        conn.commit()
        conn.close()
        logger.debug("URL 池数据库初始化完成")
    
    def _extract_domain(self, url: str) -> Optional[str]:
        """从 URL 中提取域名（规范化）"""
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                return None
            
            # 使用 tldextract 提取主域名
            extracted = self._tldextract(url)
            domain = f"{extracted.domain}.{extracted.suffix}".lower()
            return domain
        except Exception as e:
            logger.debug(f"提取域名失败 {url}: {e}")
            return None
    
    def _is_english_url(self, url: str) -> bool:
        """判断 URL 是否可能是英文内容
        
        简单启发式方法：
        1. 检查 URL 路径中是否包含 'en' 或 '/en/'
        2. 某些域名默认是英文（如 bbc.com, reuters.com）
        """
        url_lower = url.lower()
        
        # 检查路径中的语言标识
        if '/en/' in url_lower or '/en' in url_lower or url_lower.endswith('/en'):
            return True
        
        # 某些域名默认是英文
        english_default_domains = {
            'bbc.com', 'reuters.com', 'cnn.com', 'nytimes.com',
            'washingtonpost.com', 'wsj.com', 'bloomberg.com'
        }
        
        domain = self._extract_domain(url)
        if domain and any(d in domain for d in english_default_domains):
            return True
        
        # 如果 URL 中没有明显的非英文标识，假设是英文
        # 注意：这是一个简化的方法，后续可以通过实际爬取内容来验证
        non_english_indicators = ['/zh/', '/cn/', '/fr/', '/de/', '/es/', '/ja/', '/ko/']
        if not any(indicator in url_lower for indicator in non_english_indicators):
            return True
        
        return False
    
    def _is_valid_url(self, url: str) -> bool:
        """验证 URL 是否合法"""
        try:
            parsed = urlparse(url)
            return parsed.scheme in ('http', 'https') and bool(parsed.netloc)
        except Exception:
            return False
    
    def _parse_source_urls(self, source_urls_str: str) -> List[str]:
        """解析 SOURCEURLS 字段（分号分隔）"""
        if not source_urls_str or pd.isna(source_urls_str):
            return []
        
        urls = [url.strip() for url in str(source_urls_str).split(';') if url.strip()]
        return urls
    
    def _parse_sources(self, sources_str: str) -> List[str]:
        """解析 SOURCES 字段（分号分隔）"""
        if not sources_str or pd.isna(sources_str):
            return []
        
        sources = [s.strip() for s in str(sources_str).split(';') if s.strip()]
        return sources

    def _resolve_whitelisted_source(self, url: str, sources: List[str]) -> tuple[bool, Optional[str], Optional[str]]:
        domain = self._extract_domain(url)
        if not domain:
            return False, None, None

        if domain in self.whitelist_domains:
            return True, domain, self.media_domains.get(domain)

        for source in sources:
            source_domain = self._extract_domain(f"https://{source}")
            if source_domain and source_domain in self.whitelist_domains:
                return True, source_domain, self.media_domains.get(source_domain)

        return False, domain, None

    @staticmethod
    def _fallback_gkg_date(date_value: str) -> Optional[str]:
        text = str(date_value or "").strip()
        if len(text) < 8:
            return None
        raw = text[:8]
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"

    def _iter_csv_candidate_rows(self, csv_path: Path):
        row_offset = 0
        reader = pd.read_csv(
            csv_path,
            sep="\t",
            names=[
                "DATE",
                "NUMARTS",
                "COUNTS",
                "THEMES",
                "LOCATIONS",
                "PERSONS",
                "ORGANIZATIONS",
                "TONE",
                "CAMEOEVENTIDS",
                "SOURCES",
                "SOURCEURLS",
            ],
            usecols=["DATE", "THEMES", "SOURCES", "SOURCEURLS"],
            dtype=str,
            na_values=[""],
            keep_default_na=False,
            chunksize=self.gkg_parse_chunk_size,
            low_memory=False,
        )

        for chunk in reader:
            chunk = chunk.dropna(subset=["DATE"])
            chunk = chunk[chunk["DATE"].astype(str).str.strip() != ""]
            if chunk.empty:
                continue

            date_series = pd.to_datetime(chunk["DATE"], format="%Y%m%d", errors="coerce")
            if date_series.isna().any():
                fallback_mask = date_series.isna()
                date_series.loc[fallback_mask] = pd.to_datetime(
                    chunk.loc[fallback_mask, "DATE"],
                    format="%Y%m%d%H%M%S",
                    errors="coerce",
                )

            gkg_dates = date_series.dt.strftime("%Y-%m-%d")
            chunk = chunk.assign(
                gkg_date=gkg_dates.fillna(chunk["DATE"].apply(self._fallback_gkg_date))
            )

            for local_idx, row in enumerate(chunk.itertuples(index=False), start=row_offset):
                source_urls = self._parse_source_urls(getattr(row, "SOURCEURLS", ""))
                if not source_urls:
                    continue

                sources = self._parse_sources(getattr(row, "SOURCES", ""))
                themes = self._parse_sources(getattr(row, "THEMES", ""))
                themes_str = ";".join(themes) if themes else None
                gkg_date = getattr(row, "gkg_date", "") or ""
                gkg_record_id = f"{local_idx}_{getattr(row, 'DATE', '')}"

                for url in source_urls:
                    if not self._is_valid_url(url):
                        continue
                    is_whitelisted, domain, source_name = self._resolve_whitelisted_source(url, sources)
                    if not is_whitelisted or not domain:
                        continue
                    if not self._is_english_url(url):
                        continue
                    yield (
                        str(uuid.uuid4()),
                        url,
                        domain,
                        source_name,
                        gkg_date,
                        gkg_record_id,
                        themes_str,
                    )

            row_offset += len(chunk)

    def _insert_candidate_rows(self, cursor: sqlite3.Cursor, conn: sqlite3.Connection, rows: List[tuple]) -> tuple[int, int]:
        if not rows:
            return 0, 0

        before_changes = conn.total_changes
        cursor.executemany(
            """
            INSERT OR IGNORE INTO url_pool
            (id, url, domain, source_name, gkg_date, gkg_record_id, themes, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            rows,
        )
        added = conn.total_changes - before_changes
        skipped = len(rows) - added
        return int(added), int(skipped)

    def _maybe_export_url_pool_jsonl(self, cursor: sqlite3.Cursor) -> None:
        jsonl_path = self.url_pool_db.parent / "url_pool.jsonl"
        if not self.export_url_pool_jsonl:
            jsonl_path.unlink(missing_ok=True)
            return

        cursor.execute(
            """
            SELECT id, url, domain, source_name, gkg_date, gkg_record_id, themes, status, created_at, updated_at
            FROM url_pool
            ORDER BY created_at DESC
            """
        )
        all_urls = cursor.fetchall()
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in all_urls:
                record = {
                    "id": row[0],
                    "url": row[1],
                    "domain": row[2],
                    "source_name": row[3],
                    "gkg_date": row[4],
                    "gkg_record_id": row[5],
                    "themes": row[6],
                    "status": row[7],
                    "created_at": row[8],
                    "updated_at": row[9],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"已导出 URL 池 JSONL: {jsonl_path}，共 {len(all_urls)} 条记录")

    def _build_from_raw_gkg(self, csv_files: List[Path]) -> Dict[str, int]:
        logger.info(f"开始从原始 GKG CSV 构建 URL 池，处理 {len(csv_files)} 个文件")

        conn = sqlite3.connect(self.url_pool_db)
        cursor = conn.cursor()
        added_count = 0
        skipped_count = 0
        candidate_count = 0

        try:
            for csv_file in tqdm(
                csv_files,
                desc="构建 URL 池",
                unit="file",
                dynamic_ncols=True,
                mininterval=0.5,
                file=sys.stdout,
            ):
                try:
                    batch_rows: List[tuple] = []
                    for candidate in self._iter_csv_candidate_rows(csv_file):
                        batch_rows.append(candidate)
                        if len(batch_rows) >= 1000:
                            added, skipped = self._insert_candidate_rows(cursor, conn, batch_rows)
                            added_count += added
                            skipped_count += skipped
                            candidate_count += len(batch_rows)
                            batch_rows = []
                    if batch_rows:
                        added, skipped = self._insert_candidate_rows(cursor, conn, batch_rows)
                        added_count += added
                        skipped_count += skipped
                        candidate_count += len(batch_rows)

                    conn.commit()
                    if self.cleanup_raw_gkg_after_build:
                        csv_file.unlink(missing_ok=True)
                except Exception as exc:
                    logger.error(f"处理 {csv_file.name} 时发生错误: {exc}")
        finally:
            self._maybe_export_url_pool_jsonl(cursor)
            conn.commit()
            conn.close()

        logger.info(
            f"URL 池构建完成: candidates={candidate_count} "
            f"added={added_count} skipped={skipped_count}"
        )
        return {"total": candidate_count, "added": added_count, "skipped": skipped_count}

    def build(self) -> Dict[str, int]:
        """构建 URL 池。优先直接消费原始 GKG CSV，避免生成 parquet 中间文件。"""
        csv_files = list(self.raw_data_dir.glob("*.gkg.csv"))
        if csv_files:
            return self._build_from_raw_gkg(csv_files)

        logger.warning("没有找到原始 GKG CSV 文件")
        return {"total": 0, "added": 0, "skipped": 0}

    def reserve_pending_urls(self, limit: int) -> List[Dict]:
        """Reserve a batch of pending URLs and mark them in_progress."""
        batch_limit = max(1, int(limit))
        conn = sqlite3.connect(self.url_pool_db)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, url, domain, source_name, gkg_date, themes
            FROM url_pool
            WHERE status = 'pending'
            ORDER BY created_at ASC, url ASC
            LIMIT ?
            """,
            (batch_limit,),
        )
        rows = cursor.fetchall()
        if not rows:
            conn.close()
            return []

        ids = [row[0] for row in rows if row and row[0]]
        placeholders = ",".join("?" for _ in ids)
        cursor.execute(
            f"""
            UPDATE url_pool
            SET status = 'in_progress', last_error = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({placeholders})
            """,
            ids,
        )
        conn.commit()
        conn.close()

        return [
            {
                "id": row[0],
                "url": row[1],
                "domain": row[2],
                "source_name": row[3],
                "gkg_date": row[4],
                "themes": row[5],
            }
            for row in rows
        ]

    def reset_status(self, *, from_status: str, to_status: str, error: Optional[str] = None) -> int:
        """Bulk move URLs from one status to another."""
        conn = sqlite3.connect(self.url_pool_db)
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE url_pool
            SET status = ?, last_error = ?, updated_at = CURRENT_TIMESTAMP
            WHERE status = ?
            """,
            (to_status, error, from_status),
        )
        changed = cursor.rowcount or 0
        conn.commit()
        conn.close()
        return int(changed)

    def bulk_update_status_by_url(self, urls: List[str], *, status: str, error: Optional[str] = None) -> int:
        clean_urls = [str(url).strip() for url in urls if str(url).strip()]
        if not clean_urls:
            return 0

        conn = sqlite3.connect(self.url_pool_db)
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in clean_urls)
        cursor.execute(
            f"""
            UPDATE url_pool
            SET status = ?, last_error = ?, updated_at = CURRENT_TIMESTAMP
            WHERE url IN ({placeholders})
            """,
            (status, error, *clean_urls),
        )
        changed = cursor.rowcount or 0
        conn.commit()
        conn.close()
        return int(changed)

    def get_pending_urls(self, limit: Optional[int] = None, batch_size: int = 100) -> List[Dict]:
        """获取待爬取的 URL 列表"""
        conn = sqlite3.connect(self.url_pool_db)
        cursor = conn.cursor()
        
        query = "SELECT id, url, domain, source_name, gkg_date, themes FROM url_pool WHERE status = 'pending'"
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        urls = []
        for row in rows:
            urls.append({
                "id": row[0],
                "url": row[1],
                "domain": row[2],
                "source_name": row[3],
                "gkg_date": row[4],
                "themes": row[5]
            })
        
        conn.close()
        return urls
    
    def update_status(self, url_id: str, status: str, error: Optional[str] = None):
        """更新 URL 状态"""
        conn = sqlite3.connect(self.url_pool_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE url_pool 
            SET status = ?, last_error = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (status, error, url_id))
        
        conn.commit()
        conn.close()
    
    def get_statistics(self) -> Dict[str, int]:
        """获取 URL 池统计信息"""
        conn = sqlite3.connect(self.url_pool_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT status, COUNT(*) FROM url_pool GROUP BY status")
        stats = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("SELECT COUNT(*) FROM url_pool")
        total = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total": total,
            "pending": stats.get("pending", 0),
            "in_progress": stats.get("in_progress", 0),
            "success": stats.get("success", 0),
            "failed": stats.get("failed", 0)
        }
