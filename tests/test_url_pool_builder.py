"""URL 池构建模块测试"""
import pytest
import sqlite3
from pathlib import Path
import tempfile
import shutil

from src.utils.config import Config
from src.url_pool_builder.builder import URLPoolBuilder


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


def test_extract_domain():
    """测试域名提取"""
    builder = URLPoolBuilder.__new__(URLPoolBuilder)
    
    # 测试基本域名提取
    domain1 = builder._extract_domain("https://www.bbc.com/news/article")
    assert domain1 == "bbc.com"
    
    domain2 = builder._extract_domain("https://reuters.com/article")
    assert domain2 == "reuters.com"
    
    domain3 = builder._extract_domain("https://news.yahoo.com/article")
    assert domain3 == "yahoo.com"
    
    # 测试无效 URL
    domain4 = builder._extract_domain("not-a-url")
    assert domain4 is None


def test_is_english_url():
    """测试英文 URL 判断"""
    builder = URLPoolBuilder.__new__(URLPoolBuilder)
    
    # 测试包含 /en/ 的 URL
    assert builder._is_english_url("https://www.example.com/en/article") is True
    assert builder._is_english_url("https://www.example.com/article/en") is True
    
    # 测试默认英文域名
    assert builder._is_english_url("https://www.bbc.com/news/article") is True
    assert builder._is_english_url("https://www.reuters.com/article") is True
    
    # 测试非英文标识
    assert builder._is_english_url("https://www.example.com/zh/article") is False
    assert builder._is_english_url("https://www.example.com/cn/article") is False


def test_is_valid_url():
    """测试 URL 合法性验证"""
    builder = URLPoolBuilder.__new__(URLPoolBuilder)
    
    assert builder._is_valid_url("https://www.example.com/article") is True
    assert builder._is_valid_url("http://www.example.com/article") is True
    assert builder._is_valid_url("ftp://www.example.com/article") is False
    assert builder._is_valid_url("not-a-url") is False
    assert builder._is_valid_url("") is False


def test_parse_source_urls():
    """测试 SOURCEURLS 解析"""
    builder = URLPoolBuilder.__new__(URLPoolBuilder)
    
    # 单个 URL
    urls1 = builder._parse_source_urls("https://www.example.com/article1")
    assert len(urls1) == 1
    assert urls1[0] == "https://www.example.com/article1"
    
    # 多个 URL（分号分隔）
    urls2 = builder._parse_source_urls("https://www.example.com/article1;https://www.example.com/article2")
    assert len(urls2) == 2
    
    # 空字符串
    urls3 = builder._parse_source_urls("")
    assert len(urls3) == 0
    
    # None
    urls4 = builder._parse_source_urls(None)
    assert len(urls4) == 0


def test_url_pool_database(temp_dir):
    """测试 URL 池数据库操作"""
    # 创建配置
    config_dict = {
        "general": {
            "start_date": "2025-01-01",
            "end_date": "2025-01-01"
        },
        "paths": {
            "raw_data_dir": str(temp_dir),
            "processed_data_dir": str(temp_dir / "processed"),
            "url_pool_db": str(temp_dir / "url_pool.db")
        },
        "whitelist": {
            "media_domains": [
                {"name": "BBC News", "domain": "bbc.com"},
                {"name": "Reuters", "domain": "reuters.com"}
            ]
        }
    }
    
    import tomli_w
    config_file = temp_dir / "test_config.toml"
    with open(config_file, "wb") as f:
        tomli_w.dump(config_dict, f)
    
    config = Config(str(config_file))
    builder = URLPoolBuilder(config)
    
    # 测试数据库初始化
    assert builder.url_pool_db.exists()
    
    # 测试统计信息
    stats = builder.get_statistics()
    assert "total" in stats
    assert stats["total"] == 0

