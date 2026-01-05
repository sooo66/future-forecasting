import asyncio
from crawl4ai import AsyncWebCrawler

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from loguru import logger

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    ProxyConfig,
    UndetectedAdapter,
    LLMConfig
)
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from crawl4ai.content_filter_strategy import PruningContentFilter, LLMContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.proxy_strategy import RoundRobinProxyStrategy

async def main():
    run_conf = CrawlerRunConfig(
        prettiify=True,
        word_count_threshold=50,
        cache_mode=CacheMode.DISABLED,
        magic=True,
        only_text=True,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter( threshold=0.48, threshold_type="dynamic"),
            options={"ignore_links": True, "ignore_images": True},

        ),
        exclude_all_images=True,
        exclude_external_links=True,
        exclude_internal_links=True,
        exclude_social_media_links=True,
        excluded_tags=["header", "footer", "nav", "aside", "figcaption"],
        excluded_selector="[class^='video'], .recommended-articles, .related-articles, .article-footer, .site-header, .site-footer, video, frame",
        target_elements=["div.node__text--full-article", "article"],
        verbose=True
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            "https://justthenews.com/government/federal-agencies/wkdfrom-waste-reform-lee-zeldin-has-transformed-epa",
            config=run_conf,
        )
        # print(result.markdown.fit_markdown[100:][:-100])  # Print first 300 chars
        print(type(result.markdown.fit_markdown))
        r = result.markdown.fit_markdown.split("\n")
        for line in r:
            if len(line.strip()) > 0:
                print(len(line.strip()), line.strip())

if __name__ == "__main__":
    asyncio.run(main())
