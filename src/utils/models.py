"""数据模型定义"""
from dataclasses import dataclass, asdict
from typing import Optional, List, Any
from datetime import datetime
import json


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
    language: Optional[str]
    tags: Optional[List[str]]
    
    def to_dict(self) -> dict:
        """转换为字典"""
        d = asdict(self)
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
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Record":
        """从 JSON 字符串创建"""
        return cls.from_dict(json.loads(json_str))
