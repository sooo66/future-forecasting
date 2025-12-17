"""URL 池构建模块"""
import sqlite3
import uuid
import json
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Dict, Set, Optional
import tldextract
from loguru import logger
from tqdm import tqdm
import pandas as pd

from utils.config import Config


class URLPoolBuilder:
    """URL 池构建器
    
    从解析后的 GKG 数据中提取 URL，过滤白名单域名，去重并构建 URL 池。
    URL 池存储在 SQLite 数据库中，便于后续查询和更新状态。
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.processed_data_dir = config.processed_data_dir
        self.url_pool_db = config.url_pool_db
        self.whitelist_domains = config.whitelist_domains
        self.media_domains = config.media_domains
        
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
            extracted = tldextract.extract(url)
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
    
    def build(self) -> Dict[str, int]:
        """构建 URL 池"""
        parquet_files = list(self.processed_data_dir.glob("*_gkg.parquet"))
        
        if not parquet_files:
            logger.warning("没有找到解析后的 GKG 文件")
            return {"total": 0, "added": 0, "skipped": 0}
        
        logger.info(f"开始构建 URL 池，处理 {len(parquet_files)} 个文件")
        
        conn = sqlite3.connect(self.url_pool_db)
        cursor = conn.cursor()
        
        total_urls = 0
        added_count = 0
        skipped_count = 0
        
        for parquet_file in tqdm(parquet_files, desc="构建 URL 池"):
            try:
                df = pd.read_parquet(parquet_file)
                
                for _, row in df.iterrows():
                    # 解析 SOURCEURLS
                    source_urls = self._parse_source_urls(row.get('SOURCEURLS', ''))
                    
                    # 解析 SOURCES（用于匹配域名）
                    sources = self._parse_sources(row.get('SOURCES', ''))
                    
                    # 解析主题（用于后续生成 tags）
                    # 注意：实际字段名是 THEMES（全大写）
                    themes = self._parse_sources(row.get('THEMES', ''))
                    themes_str = ';'.join(themes) if themes else None
                    
                    gkg_date = row.get('gkg_date', '')
                    # 支持两种字段名格式
                    gkg_record_id = row.get('gkg_record_id', '') or row.get('GKGRECORDID', '')
                    
                    # 处理每个 URL
                    for url in source_urls:
                        if not self._is_valid_url(url):
                            continue
                        
                        domain = self._extract_domain(url)
                        if not domain:
                            continue
                        
                        # 检查域名是否在白名单中
                        # 同时检查 SOURCES 字段中是否包含白名单域名
                        is_whitelisted = False
                        source_name = None
                        
                        if domain in self.whitelist_domains:
                            is_whitelisted = True
                            source_name = self.media_domains.get(domain)
                        else:
                            # 检查 SOURCES 字段
                            for source in sources:
                                source_domain = self._extract_domain(f"https://{source}")
                                if source_domain and source_domain in self.whitelist_domains:
                                    is_whitelisted = True
                                    source_name = self.media_domains.get(source_domain)
                                    domain = source_domain  # 使用匹配的域名
                                    break
                        
                        if not is_whitelisted:
                            continue
                        
                        # 检查是否是英文内容
                        if not self._is_english_url(url):
                            continue
                        
                        # 检查 URL 是否已存在
                        cursor.execute("SELECT id FROM url_pool WHERE url = ?", (url,))
                        if cursor.fetchone():
                            skipped_count += 1
                            continue
                        
                        # 插入新 URL
                        url_id = str(uuid.uuid4())
                        cursor.execute("""
                            INSERT INTO url_pool 
                            (id, url, domain, source_name, gkg_date, gkg_record_id, themes, status)
                            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
                        """, (url_id, url, domain, source_name, gkg_date, gkg_record_id, themes_str))
                        
                        added_count += 1
                        total_urls += 1
                
            except Exception as e:
                logger.error(f"处理 {parquet_file.name} 时发生错误: {e}")
        
        conn.commit()
        
        # 保存完整的 URL 池到 JSONL 文件（包含所有 URL，不仅仅是本次新增的）
        jsonl_path = self.url_pool_db.parent / "url_pool.jsonl"
        cursor.execute("""
            SELECT id, url, domain, source_name, gkg_date, gkg_record_id, themes, status, created_at, updated_at
            FROM url_pool
            ORDER BY created_at DESC
        """)
        all_urls = cursor.fetchall()
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
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
                    "updated_at": row[9]
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"已保存完整 URL 池到 JSONL 文件: {jsonl_path}，共 {len(all_urls)} 条记录")
        
        conn.close()
        
        logger.info(f"URL 池构建完成: 新增 {added_count} 个 URL，跳过 {skipped_count} 个重复 URL，总计 {total_urls} 个 URL")
        
        return {
            "total": total_urls,
            "added": added_count,
            "skipped": skipped_count
        }
    
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
            "success": stats.get("success", 0),
            "failed": stats.get("failed", 0)
        }

