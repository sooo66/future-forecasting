"""爬虫模块测试"""
import pytest
from pathlib import Path
import tempfile
import shutil

from src.crawler.extractor import ContentExtractor


@pytest.fixture
def sample_html():
    """示例 HTML"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Test Article Title</title>
        <meta property="og:title" content="OG Title">
        <meta name="description" content="This is a test article description">
        <meta property="article:published_time" content="2025-01-01T12:00:00Z">
    </head>
    <body>
        <article>
            <h1>Article Heading</h1>
            <p>This is the first paragraph of the article.</p>
            <p>This is the second paragraph of the article.</p>
        </article>
    </body>
    </html>
    """


def test_extract_title(sample_html):
    """测试标题提取"""
    extractor = ContentExtractor()
    
    title = extractor.extract_title(sample_html, "https://www.example.com/article")
    
    # 应该优先使用 og:title
    assert title == "OG Title"


def test_extract_summary(sample_html):
    """测试摘要提取"""
    extractor = ContentExtractor()
    
    summary = extractor.extract_summary(sample_html)
    
    assert summary == "This is a test article description"


def test_extract_published_at(sample_html):
    """测试发布时间提取"""
    extractor = ContentExtractor()
    
    published_at = extractor.extract_published_at(sample_html)
    
    assert published_at == "2025-01-01T12:00:00Z"


def test_extract_content(sample_html):
    """测试内容提取"""
    extractor = ContentExtractor()
    
    content = extractor.extract_content(sample_html)
    
    assert "Article Heading" in content
    assert "first paragraph" in content
    assert "second paragraph" in content


def test_extract_language(sample_html):
    """测试语言提取"""
    extractor = ContentExtractor()
    
    language = extractor.extract_language(sample_html, "https://www.example.com/article")
    
    assert language == "en"

def test_extract_content_prefers_main_body_over_nav():
    """markdown 只有导航时应回退到 HTML 主体"""
    extractor = ContentExtractor()
    noisy_markdown = "[Home](/home) [Subscribe](/subscribe)"
    html = """
    <html>
        <body>
            <nav>Home Subscribe</nav>
            <article>
                <p>Main body text is here with enough words to be selected as meaningful content.</p>
                <p>Second paragraph adds more context for the article body.</p>
            </article>
        </body>
    </html>
    """
    content = extractor.extract_content(html, markdown=noisy_markdown, url="https://example.com/news")
    
    assert "Main body text" in content
    assert "Subscribe" not in content


def test_extract_tags():
    """测试分类标签提取"""
    extractor = ContentExtractor()
    
    # themes 分类
    tags1 = extractor.extract_tags("ECON;POLITICS;SPORTS", "https://www.example.com/article")
    assert tags1 is not None
    assert "business_finance" in tags1
    assert "politics" in tags1
    assert "sports" in tags1
    
    # URL 分类
    tags2 = extractor.extract_tags(None, "https://www.example.com/technology/article")
    assert tags2 is not None
    assert "technology" in tags2
    
    # 无匹配
    tags3 = extractor.extract_tags(None, "https://www.example.com/article")
    assert tags3 is None or len(tags3) == 0
