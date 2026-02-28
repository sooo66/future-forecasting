# -*- coding: utf-8 -*-
"""Twitter/X importer for public tweets."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional
import xml.etree.ElementTree as ET
from uuid import uuid4

from dateutil import parser as date_parser
from loguru import logger

from utils.importer_base import BaseImporter
from utils.models import Record

from .twitter_client import TwitterClient, extract_tweet_id
from .twitter_types import TweetData

FIXED_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


DEFAULT_TWITTER_ACCOUNTS = [
    "BBCWorld",
    "Reuters",
    "AP",
    "npr",
    "theeconomist",
    "FT",
    "business",
    "WSJ",
    "nytimes",
    "guardian",
    "washingtonpost",
    "Bloomberg",
    "CNBC",
    "TechCrunch",
    "verge",
    "WIRED",
    "ArsTechnica",
    "MITTechReview",
    "Nature",
    "sciencemagazine",
    "newscientist",
    "NASA",
    "SpaceX",
    "WHO",
    "CDCDirector",
    "UN",
    "WorldBank",
    "IMFNews",
    "WEF",
    "OECD",
    "OurWorldInData",
    "Kaggle",
    "OpenAI",
    "GoogleAI",
    "DeepMind",
    "AnthropicAI",
    "OpenBB_FIN",
    "GitHub",
    "HuggingFace",
    "zacharylipton",
    "ylecun",
    "AndrewYNg",
    "ianbremmer",
    "ericgeller",
    "alexstamos",
    "briankrebs",
    "MatthewHerper",
    "EricTopol",
    "karaswisher",
    "sama",
    "pmarca",
    "naval",
]

DEFAULT_RSSHUB_INSTANCES = [
    "https://rsshub.app",
]


@dataclass(frozen=True)
class TwitterConfig:
    start_date: str
    end_date: str
    output_path: str
    per_day_limit: int = 300
    accounts: Optional[List[str]] = None
    rsshub_instances: Optional[List[str]] = None


class TwitterImportError(RuntimeError):
    pass


class TwitterImporter(BaseImporter):
    def __init__(self, config: TwitterConfig) -> None:
        super().__init__(timeout=20, retries=0, backoff=0.0, user_agent=FIXED_USER_AGENT)
        self.config = config
        self.client = TwitterClient()

    def discover_urls(self) -> List[str]:
        start_dt = BaseImporter._parse_datetime(self.config.start_date)
        end_dt = BaseImporter._parse_datetime(self.config.end_date)
        if not start_dt or not end_dt:
            raise TwitterImportError("Invalid date range")
        if start_dt > end_dt:
            raise TwitterImportError("start_date must be <= end_date")

        accounts = self.config.accounts or DEFAULT_TWITTER_ACCOUNTS
        instances = self.config.rsshub_instances or DEFAULT_RSSHUB_INSTANCES
        urls: List[str] = []
        seen = set()

        for account in accounts:
            found_for_account = False
            for instance in instances:
                feed_url = f"{instance.rstrip('/')}/twitter/user/{account}"
                try:
                    xml_text = self._request_text(feed_url, accept="application/rss+xml")
                except RuntimeError:
                    continue

                items = _parse_rss_items(xml_text)
                for link, pub_dt in items:
                    if not link or not pub_dt:
                        continue
                    if not _in_range_dt(pub_dt, start_dt, end_dt):
                        continue
                    if link in seen:
                        continue
                    seen.add(link)
                    urls.append(link)
                    found_for_account = True
                    if len(urls) >= self.config.per_day_limit:
                        return urls
                break

            if not found_for_account:
                logger.warning(f"No RSS items for @{account}")
            if len(urls) >= self.config.per_day_limit:
                break

        logger.info(f"Discovered {len(urls)} tweet URLs")
        return urls

    def run(self) -> List[Record]:
        return collect_records(self)

    def fetch_record(self, url: str) -> Optional[Record]:
        try:
            tweet_id = extract_tweet_id(url)
            tweet = self.client.get_tweet(tweet_id)
        except Exception as exc:
            logger.warning(f"Tweet fetch failed: {url} ({exc})")
            return None

        normalized_url = f"https://x.com/i/web/status/{tweet_id}"

        pubtime = _parse_twitter_datetime(tweet.created_at)
        if not pubtime:
            return None
        if not _in_range(pubtime, self.config.start_date, self.config.end_date):
            return None

        content = _build_tweet_content(tweet)
        title = _truncate_title(tweet)
        tags = _build_tags(tweet)

        return Record(
            id=str(uuid4()),
            source="sociomedia/twitter",
            url=normalized_url,
            title=title,
            description=None,
            content=content,
            published_at=pubtime,
            pubtime=pubtime,
            language="en",
            tags=tags or None,
        )


def _build_tweet_content(tweet: TweetData) -> str:
    text = tweet.full_text or tweet.text or ""
    text = _clean_text(text)
    parts = [p for p in text.split("\n") if p.strip()]

    if tweet.quoted_tweet:
        quoted_text = tweet.quoted_tweet.full_text or tweet.quoted_tweet.text
        quoted_text = _clean_text(quoted_text)
        if quoted_text:
            parts.append(f"[quoted] {quoted_text}")

    for media in tweet.media:
        if media.media_url:
            parts.append(media.media_url)
        if media.video_url:
            parts.append(media.video_url)

    return "\n".join(parts).strip()


def _build_tags(tweet: TweetData) -> List[str]:
    tags = ["twitter"]
    if tweet.author and tweet.author.screen_name:
        tags.append(f"@{tweet.author.screen_name}")
    return tags


def _truncate_title(tweet: TweetData) -> str:
    raw = tweet.full_text or tweet.text or ""
    raw = _clean_text(raw)
    if not raw:
        return f"Tweet {tweet.id}"
    return raw[:100]


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s*https://t\.co/\w+\s*$", "", text)
    text = re.sub(r"https://t\.co/\w+", "", text)
    return text.strip()


def _parse_twitter_datetime(value: str) -> Optional[str]:
    if not value:
        return None
    try:
        dt = datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")
    except ValueError:
        try:
            dt = date_parser.parse(value)
        except (ValueError, TypeError):
            return None
    return dt.astimezone(timezone.utc).isoformat()


def _in_range(pubtime_iso: str, start_date: str, end_date: str) -> bool:
    start_dt = BaseImporter._parse_datetime(start_date)
    end_dt = BaseImporter._parse_datetime(end_date)
    pub_dt = BaseImporter._parse_datetime(pubtime_iso)
    if not start_dt or not end_dt or not pub_dt:
        return False
    return start_dt <= pub_dt <= end_dt


def _in_range_dt(dt: datetime, start_dt: datetime, end_dt: datetime) -> bool:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return start_dt <= dt.astimezone(timezone.utc) <= end_dt


def _parse_rss_items(xml_text: str) -> List[tuple[str, Optional[datetime]]]:
    items: List[tuple[str, Optional[datetime]]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items

    channel = root.find("channel")
    if channel is None:
        return items

    for item in channel.findall("item"):
        link = item.findtext("link") or ""
        pub_date = item.findtext("pubDate") or ""
        pub_dt = None
        if pub_date:
            try:
                pub_dt = date_parser.parse(pub_date)
            except (ValueError, TypeError):
                pub_dt = None
        items.append((link.strip(), pub_dt))

    return items


def collect_records(importer: TwitterImporter) -> List[Record]:
    urls = importer.discover_urls()
    if not urls:
        return []
    records: List[Record] = []
    for url in urls:
        record = importer.fetch_record(url)
        if record:
            records.append(record)
        if len(records) >= importer.config.per_day_limit:
            break
    return records
