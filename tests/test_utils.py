"""工具函数测试"""
import pytest
from pathlib import Path
import tempfile
import shutil

from src.utils.models import Record


def test_record_serialization():
    """测试 Record 序列化"""
    record = Record(
        id="test-id",
        source="news/bbc.com",
        url="https://www.bbc.com/news/article",
        title="Test Article",
        summary="Test summary",
        content="Test content",
        published_at="2025-01-01T12:00:00Z",
        language="en",
        tags=["politics", "news"]
    )
    
    # 测试 to_dict
    record_dict = record.to_dict()
    assert record_dict["id"] == "test-id"
    assert record_dict["source"] == "news/bbc.com"
    assert record_dict["tags"] == ["politics", "news"]
    
    # 测试 to_json
    json_str = record.to_json()
    assert "test-id" in json_str
    assert "news/bbc.com" in json_str
    
    # 测试 from_json
    record2 = Record.from_json(json_str)
    assert record2.id == record.id
    assert record2.source == record.source
    assert record2.tags == record.tags


def test_record_with_optional_fields():
    """测试 Record 可选字段"""
    record = Record(
        id="test-id",
        source="news/bbc.com",
        url="https://www.bbc.com/news/article",
        title="Test Article",
        summary=None,
        content="Test content",
        published_at="2025-01-01T12:00:00Z",
        language=None,
        tags=None
    )
    
    record_dict = record.to_dict()
    assert record_dict["summary"] is None
    assert record_dict["language"] is None
    assert record_dict["tags"] is None

