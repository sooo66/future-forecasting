"""GDELT 下载和解析模块测试"""
import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

from src.utils.config import Config
from src.gdelt_downloader.parser import GDELTParser


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_gkg_data():
    """示例 GKG 数据（制表符分隔）"""
    # GKG 格式示例（简化版）
    return """GKGRECORDID\tDATE\tSourceCollectionIdentifier\tSourceCommonName\tDocumentIdentifier\tCounts\tV2Counts\tThemes\tV2Themes\tLocations\tV2Locations\tPersons\tV2Persons\tOrganizations\tV2Organizations\tV2Tone\tDates\tV2Dates\tGCAM\tSharingImage\tRelatedImages\tSocialImageEmbeds\tSocialVideoEmbeds\tQuotations\tV2Quotations\tAllNames\tAmounts\tV2Amounts\tTranslationInfo\tExtras\tSOURCES\tSOURCEURLS
1234567890\t20250101120000\t1\tBBC\tDOC123\t\t\tECON\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tbbc.com\thttps://www.bbc.com/news/article1
9876543210\t20250101130000\t1\tReuters\tDOC456\t\t\tPOLITICS\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\treuters.com\thttps://www.reuters.com/article2
"""


def test_parse_gkg_file(temp_dir, sample_gkg_data):
    """测试 GKG 文件解析"""
    # 创建测试文件
    csv_file = temp_dir / "20250101.gkg.csv"
    csv_file.write_text(sample_gkg_data)
    
    # 创建配置
    config_dict = {
        "general": {
            "start_date": "2025-01-01",
            "end_date": "2025-01-01"
        },
        "paths": {
            "raw_data_dir": str(temp_dir),
            "processed_data_dir": str(temp_dir / "processed")
        }
    }
    
    # 创建临时配置文件
    import tomli_w
    config_file = temp_dir / "test_config.toml"
    with open(config_file, "wb") as f:
        tomli_w.dump(config_dict, f)
    
    config = Config(str(config_file))
    parser = GDELTParser(config)
    
    # 解析文件
    df = parser._parse_gkg_file(csv_file)
    
    assert df is not None
    assert len(df) == 2
    assert "GKGRECORDID" in df.columns
    assert "DATE" in df.columns
    assert "SOURCEURLS" in df.columns
    assert "SOURCES" in df.columns
    assert "Themes" in df.columns


def test_extract_urls_from_gkg():
    """测试从 GKG 数据中提取 URL"""
    # 模拟 GKG 数据
    data = {
        "GKGRECORDID": ["123", "456"],
        "SOURCEURLS": [
            "https://www.bbc.com/news/article1",
            "https://www.reuters.com/article2;https://www.cnn.com/article3"
        ],
        "SOURCES": ["bbc.com", "reuters.com;cnn.com"],
        "Themes": ["ECON", "POLITICS"],
        "DATE": pd.to_datetime(["2025-01-01", "2025-01-01"])
    }
    df = pd.DataFrame(data)
    
    # 验证 URL 提取逻辑
    from src.url_pool_builder.builder import URLPoolBuilder
    
    # 这里只测试 URL 解析逻辑，不测试完整的构建流程
    builder = URLPoolBuilder.__new__(URLPoolBuilder)  # 不调用 __init__
    
    urls1 = builder._parse_source_urls("https://www.bbc.com/news/article1")
    assert len(urls1) == 1
    assert urls1[0] == "https://www.bbc.com/news/article1"
    
    urls2 = builder._parse_source_urls("https://www.reuters.com/article2;https://www.cnn.com/article3")
    assert len(urls2) == 2
    assert "https://www.reuters.com/article2" in urls2
    assert "https://www.cnn.com/article3" in urls2

