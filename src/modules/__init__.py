"""Module registry for unified info/kb pipeline."""
from __future__ import annotations

from modules.info.news import NewsModule
from modules.info.substack import SubstackModule
from modules.info.reddit import RedditModule
from modules.info.arxiv import ArxivModule
from modules.kb.openstax import OpenStaxModule
from modules.kb.worldbank import WorldBankModule


MODULE_REGISTRY = {
    "info.news": NewsModule,
    "info.blog.substack": SubstackModule,
    "info.sociomedia.reddit": RedditModule,
    "info.paper.arxiv": ArxivModule,
    "kb.book.openstax": OpenStaxModule,
    "kb.report.world_bank": WorldBankModule,
}

DEFAULT_MODULES = [
    "info.news",
    "info.blog.substack",
    "info.sociomedia.reddit",
    "info.paper.arxiv",
    "kb.book.openstax",
    "kb.report.world_bank",
]

