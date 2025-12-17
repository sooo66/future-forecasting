"""Crawl4AI 爬虫模块"""

from .extractor import ContentExtractor

# 懒加载 NewsCrawler，避免在未安装 crawl4ai 时阻塞 ContentExtractor 的使用/测试
try:
    from .crawler import NewsCrawler  # type: ignore
    _crawler_import_error = None
except Exception as exc:  # pragma: no cover - 仅在依赖缺失时触发
    NewsCrawler = None  # type: ignore
    _crawler_import_error = exc

__all__ = ["NewsCrawler", "ContentExtractor"]


def __getattr__(name):
    if name == "NewsCrawler" and NewsCrawler is None:
        raise ImportError("NewsCrawler 需要安装 crawl4ai 依赖") from _crawler_import_error
    raise AttributeError(f"module {__name__} has no attribute {name}")
