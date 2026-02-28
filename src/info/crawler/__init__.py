"""爬虫模块入口"""

from .extractor import Extractor
from .crawler import Crawler

# 向后兼容旧名称 ProgressiveEvasionCrawler
ProgressiveEvasionCrawler = Crawler

__all__ = [
    "Extractor",
    "Crawler",
    "ProgressiveEvasionCrawler",
]
