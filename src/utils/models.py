"""数据模型定义"""
from dataclasses import dataclass, asdict, fields
from typing import Optional, Any
import json

from .time_utils import to_day, to_iso_utc


@dataclass
class Record:
    """新闻记录数据模型"""
    id: str  # uuid4
    source: str  # 如 "news/bbc.com"
    url: str
    title: str
    description: Optional[str]
    content: Any  # 修改为 Any，因为可能是 MarkdownGenerationResult 或字符串
    published_at: str  # ISO8601 字符串
    language: Optional[str] = None
    pubtime: Optional[str] = None  # 与 published_at 对齐的发布时间
    timestamp: Optional[str] = None  # YYYY-MM-DD，供检索阶段做时间过滤
    record_type: str = "info"  # info / kb / tool / question
    metadata: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """转换为字典"""
        d = asdict(self)
        # 统一时间字段：优先使用 published_at / pubtime 推导日级时间戳
        canonical_ts = to_iso_utc(self.published_at) or to_iso_utc(self.pubtime) or self.published_at or self.pubtime
        if canonical_ts:
            d["published_at"] = canonical_ts
        if d.get("timestamp"):
            d["timestamp"] = to_day(d.get("timestamp")) or d.get("timestamp")
        else:
            d["timestamp"] = to_day(canonical_ts) or to_day(self.published_at) or to_day(self.pubtime)

        # 处理 MarkdownGenerationResult 或其他非基本类型
        if hasattr(self.content, "model_dump"): # Crawl4AI uses Pydantic
             d["content"] = self.content.model_dump()
        elif hasattr(self.content, "__dict__") and not isinstance(self.content, (str, dict, list, int, float, bool, type(None))):
             try:
                 d["content"] = asdict(self.content)
             except:
                 d["content"] = str(self.content)
        return d
    
    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Record":
        """从字典创建"""
        allowed = {f.name for f in fields(cls)}
        payload = {k: v for k, v in data.items() if k in allowed}
        return cls(**payload)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Record":
        """从 JSON 字符串创建"""
        return cls.from_dict(json.loads(json_str))
