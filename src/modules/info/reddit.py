"""Reddit module with date-ranged crawling and resume support."""
from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Iterable, Iterator, List, Optional

import requests
from loguru import logger

from core.contracts import TextRecord, stable_record_id
from modules.base import RunContext
from modules.common.proxy_pool import configure_requests_session
from utils.time_utils import to_day


DEFAULT_SUBREDDITS = [
    "worldnews",
    "internationalnews",
    "geopolitics",
    "worldpolitics",
    "ukraine",
    "middleeast",
    "war",
    "europe",
    "china",
    "india",
    "news",
    "politics",
    "economics",
    "economy",
    "business",
    "finance",
    "investing",
    "energy",
    "renewableenergy",
    "climate",
    "climatechange",
    "science",
    "technology",
    "futurology",
    "singularity",
    "space",
    "ukpolitics",
    "australia",
    "neutralpolitics",
    "worldevents",
    "inthenews",
    "supplychain",
]

PULLPUSH_API_URL = "https://api.pullpush.io/reddit/search/submission/"


class RedditModule:
    name = "info.sociomedia.reddit"

    def __init__(
        self,
        *,
        subreddits: Optional[List[str]] = None,
        max_comments_per_post: int = 3,
        use_pullpush: bool = True,
        pullpush_page_size: int = 100,
        pullpush_max_pages: int = 200,
        listing_page_cap: int = 300,
        listing_sleep_sec: float = 0.2,
        min_records_per_subreddit: int = 40,
        pullpush_stale_days: int = 45,
        enable_top_fallback: bool = True,
        enable_controversial_fallback: bool = True,
        top_page_cap: int = 80,
        controversial_page_cap: int = 80,
        top_nohit_page_cap: int = 8,
        progress_log_every: int = 25,
        request_connect_timeout_sec: float = 8.0,
        request_read_timeout_sec: float = 40.0,
        comments_connect_timeout_sec: float = 3.0,
        comments_read_timeout_sec: float = 6.0,
        listing_retry_max: int = 3,
        comments_retry_max: int = 1,
        max_comment_posts_per_subreddit: int = 25,
        top_fallback_subreddits: Optional[List[str]] = None,
    ) -> None:
        self.subreddits = subreddits or list(DEFAULT_SUBREDDITS)
        self.max_comments_per_post = max(0, max_comments_per_post)
        self.use_pullpush = bool(use_pullpush)
        self.pullpush_page_size = max(10, min(100, int(pullpush_page_size)))
        self.pullpush_max_pages = max(1, int(pullpush_max_pages))
        self.listing_page_cap = max(1, int(listing_page_cap))
        self.listing_sleep_sec = max(0.0, float(listing_sleep_sec))
        self.min_records_per_subreddit = max(0, int(min_records_per_subreddit))
        self.pullpush_stale_days = max(0, int(pullpush_stale_days))
        self.enable_top_fallback = bool(enable_top_fallback)
        self.enable_controversial_fallback = bool(enable_controversial_fallback)
        self.top_page_cap = max(1, int(top_page_cap))
        self.controversial_page_cap = max(1, int(controversial_page_cap))
        self.top_nohit_page_cap = max(1, int(top_nohit_page_cap))
        self.progress_log_every = max(1, int(progress_log_every))
        self.request_timeout = (
            max(1.0, float(request_connect_timeout_sec)),
            max(2.0, float(request_read_timeout_sec)),
        )
        self.comments_timeout = (
            max(1.0, float(comments_connect_timeout_sec)),
            max(2.0, float(comments_read_timeout_sec)),
        )
        self.listing_retry_max = max(1, int(listing_retry_max))
        self.comments_retry_max = max(1, int(comments_retry_max))
        self.max_comment_posts_per_subreddit = max(0, int(max_comment_posts_per_subreddit))
        default_top_fallback_subs = {s.strip().lower() for s in self.subreddits if s and s.strip()}
        if top_fallback_subreddits is None:
            self.top_fallback_subreddits = default_top_fallback_subs
        else:
            self.top_fallback_subreddits = {s.strip().lower() for s in top_fallback_subreddits if s and s.strip()}
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "future-forecasting-reddit-module/1.0",
            }
        )
        self._proxy_enabled = configure_requests_session(self.session)
        self._proxy_bypassed = False
        self._request_stats = {
            "proxy_bypass": 0,
            "pullpush_failures": 0,
            "listing_failures": 0,
            "comment_failures": 0,
            "http_429": 0,
            "http_4xx": 0,
        }

    def run(self, ctx: RunContext) -> Iterable[dict]:
        date_from = ctx.date_from.astimezone(timezone.utc)
        date_to = ctx.date_to.astimezone(timezone.utc)
        from_ts = int(date_from.timestamp())
        to_ts = int(date_to.timestamp())

        logger.info(f"[{self.name}] crawling reddit from={date_from.date()} to={date_to.date()}")
        def _iter():
            count = 0
            for sub in self.subreddits:
                for row in self._crawl_subreddit(sub, from_ts=from_ts, to_ts=to_ts):
                    count += 1
                    yield row
            logger.info(f"[{self.name}] normalized records={count}")
            if count == 0:
                logger.warning(f"[{self.name}] no records emitted; request_stats={self._request_stats}")

        return _iter()

    def _crawl_subreddit(self, subreddit: str, *, from_ts: int, to_ts: int) -> Iterator[dict]:
        seen_post_ids: set[str] = set()
        emitted_count = 0
        comment_budget = self.max_comment_posts_per_subreddit

        pullpush_rows: List[dict] = []
        pullpush_latest_ts = 0
        if self.use_pullpush:
            pullpush_rows = self._fetch_pullpush_submissions(subreddit, from_ts=from_ts, to_ts=to_ts)
            logger.info(f"[{self.name}] subreddit={subreddit} pullpush_rows={len(pullpush_rows)}")

        for post in pullpush_rows:
            created_utc = self._safe_int(post.get("created_utc"))
            if created_utc is None:
                continue
            if created_utc > pullpush_latest_ts:
                pullpush_latest_ts = created_utc
            if created_utc < from_ts or created_utc > to_ts:
                continue
            post_id = self._normalize_post_id(post)
            if post_id and post_id in seen_post_ids:
                continue
            fetch_comments = comment_budget > 0
            record = self._normalize_post(
                post,
                subreddit=subreddit,
                created_utc=created_utc,
                fetch_comments=fetch_comments,
            )
            if record:
                emitted_count += 1
                if emitted_count % self.progress_log_every == 0:
                    logger.info(f"[{self.name}] subreddit={subreddit} emitted={emitted_count}")
                yield record
                if fetch_comments and ((record.get("payload") or {}).get("num_comments") or 0) > 0:
                    comment_budget -= 1
                if post_id:
                    seen_post_ids.add(post_id)

        stale_cutoff = to_ts - (self.pullpush_stale_days * 86400)
        pullpush_stale = bool(pullpush_latest_ts and pullpush_latest_ts < stale_cutoff)
        pulled_count = len(seen_post_ids)
        need_listing = (not pulled_count) or pullpush_stale or (pulled_count < self.min_records_per_subreddit)

        # PullPush 数据可能滞后；优先补抓 new listing。
        if need_listing:
            listing_rows = self._fetch_listing_range(subreddit, from_ts=from_ts, to_ts=to_ts)
            logger.info(f"[{self.name}] subreddit={subreddit} listing_rows={len(listing_rows)}")
            for post in listing_rows:
                created_utc = self._safe_int(post.get("created_utc"))
                if created_utc is None:
                    continue
                post_id = self._normalize_post_id(post)
                if post_id and post_id in seen_post_ids:
                    continue
                fetch_comments = comment_budget > 0
                record = self._normalize_post(
                    post,
                    subreddit=subreddit,
                    created_utc=created_utc,
                    fetch_comments=fetch_comments,
                )
                if record:
                    emitted_count += 1
                    if emitted_count % self.progress_log_every == 0:
                        logger.info(f"[{self.name}] subreddit={subreddit} emitted={emitted_count}")
                    yield record
                    if fetch_comments and ((record.get("payload") or {}).get("num_comments") or 0) > 0:
                        comment_budget -= 1
                    if post_id:
                        seen_post_ids.add(post_id)

        # 对历史窗口补抓 ranked listing（top / controversial），提升覆盖。
        if len(seen_post_ids) < self.min_records_per_subreddit and subreddit.strip().lower() in self.top_fallback_subreddits:
            ranked_plans: list[tuple[str, str, int]] = []
            if self.enable_top_fallback:
                ranked_plans.extend([("top", "year", self.top_page_cap), ("top", "all", self.top_page_cap)])
            if self.enable_controversial_fallback:
                ranked_plans.extend(
                    [
                        ("controversial", "year", self.controversial_page_cap),
                        ("controversial", "all", self.controversial_page_cap),
                    ]
                )
            for listing_kind, period, page_cap in ranked_plans:
                rows = self._fetch_ranked_range(
                    subreddit,
                    from_ts=from_ts,
                    to_ts=to_ts,
                    listing_kind=listing_kind,
                    time_filter=period,
                    page_cap=page_cap,
                )
                logger.info(
                    f"[{self.name}] subreddit={subreddit} {listing_kind}_rows={len(rows)} period={period}"
                )
                for post in rows:
                    created_utc = self._safe_int(post.get("created_utc"))
                    if created_utc is None:
                        continue
                    post_id = self._normalize_post_id(post)
                    if post_id and post_id in seen_post_ids:
                        continue
                    fetch_comments = comment_budget > 0
                    record = self._normalize_post(
                        post,
                        subreddit=subreddit,
                        created_utc=created_utc,
                        fetch_comments=fetch_comments,
                    )
                    if record:
                        emitted_count += 1
                        if emitted_count % self.progress_log_every == 0:
                            logger.info(f"[{self.name}] subreddit={subreddit} emitted={emitted_count}")
                        yield record
                        if fetch_comments and ((record.get("payload") or {}).get("num_comments") or 0) > 0:
                            comment_budget -= 1
                        if post_id:
                            seen_post_ids.add(post_id)
        logger.info(f"[{self.name}] subreddit={subreddit} done emitted={emitted_count}")

    def _fetch_pullpush_submissions(self, subreddit: str, *, from_ts: int, to_ts: int) -> List[dict]:
        rows: List[dict] = []
        before_cursor = to_ts
        page = 0

        while page < self.pullpush_max_pages and before_cursor >= from_ts:
            params = {
                "subreddit": subreddit,
                "after": from_ts,
                "before": before_cursor,
                "size": self.pullpush_page_size,
                "sort": "desc",
                "sort_type": "created_utc",
            }
            try:
                resp = self._request(
                    PULLPUSH_API_URL,
                    params=params,
                    timeout=self.request_timeout,
                    failure_key="pullpush_failures",
                )
            except Exception as exc:
                logger.warning(f"[{self.name}] pullpush request failed subreddit={subreddit}: {exc}")
                break
            if resp.status_code >= 400:
                if resp.status_code == 429:
                    self._request_stats["http_429"] += 1
                else:
                    self._request_stats["http_4xx"] += 1
                logger.warning(
                    f"[{self.name}] pullpush status={resp.status_code} subreddit={subreddit} page={page}"
                )
                break
            try:
                payload = resp.json()
            except Exception:
                break
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, list) or not data:
                break

            oldest_seen = before_cursor
            for post in data:
                if not isinstance(post, dict):
                    continue
                created_utc = self._safe_int(post.get("created_utc"))
                if created_utc is None:
                    continue
                if created_utc < from_ts:
                    continue
                rows.append(post)
                if created_utc < oldest_seen:
                    oldest_seen = created_utc

            page += 1
            if len(data) < self.pullpush_page_size:
                break
            next_before = oldest_seen - 1
            if next_before >= before_cursor:
                break
            before_cursor = next_before
            time.sleep(0.05)

        return rows

    def _fetch_listing_range(self, subreddit: str, *, from_ts: int, to_ts: int) -> List[dict]:
        after: Optional[str] = None
        rows_in_range: List[dict] = []
        page = 0
        stop = False

        while not stop and page < self.listing_page_cap:
            rows, after = self._fetch_listing(subreddit, after=after)
            page += 1
            if not rows:
                break

            min_created = None
            for post in rows:
                created_utc = self._safe_int(post.get("created_utc"))
                if created_utc is None:
                    continue
                if min_created is None or created_utc < min_created:
                    min_created = created_utc
                if created_utc < from_ts or created_utc > to_ts:
                    continue
                rows_in_range.append(post)

            if min_created is not None and min_created < from_ts:
                stop = True
            if not after:
                break
            time.sleep(self.listing_sleep_sec)

        return rows_in_range

    def _fetch_ranked_range(
        self,
        subreddit: str,
        *,
        from_ts: int,
        to_ts: int,
        listing_kind: str,
        time_filter: str,
        page_cap: int,
    ) -> List[dict]:
        after: Optional[str] = None
        rows_in_range: List[dict] = []
        page = 0
        nohit_pages = 0
        while page < page_cap:
            rows, after = self._fetch_ranked_listing(
                subreddit,
                listing_kind=listing_kind,
                after=after,
                time_filter=time_filter,
            )
            page += 1
            if not rows:
                break
            page_hits = 0
            for post in rows:
                created_utc = self._safe_int(post.get("created_utc"))
                if created_utc is None:
                    continue
                if created_utc < from_ts or created_utc > to_ts:
                    continue
                rows_in_range.append(post)
                page_hits += 1
            if page_hits == 0:
                nohit_pages += 1
                if nohit_pages >= self.top_nohit_page_cap:
                    break
            else:
                nohit_pages = 0
            if not after:
                break
            time.sleep(self.listing_sleep_sec)
        return rows_in_range

    def _fetch_listing(self, subreddit: str, *, after: Optional[str]) -> tuple[List[dict], Optional[str]]:
        url = f"https://www.reddit.com/r/{subreddit}/new.json"
        params = {"limit": 100, "raw_json": 1}
        if after:
            params["after"] = after
        return self._fetch_listing_payload(url, params=params)

    def _fetch_ranked_listing(
        self,
        subreddit: str,
        *,
        listing_kind: str,
        after: Optional[str],
        time_filter: str,
    ) -> tuple[List[dict], Optional[str]]:
        kind = str(listing_kind or "top").strip().lower()
        if kind not in {"top", "controversial"}:
            kind = "top"
        url = f"https://www.reddit.com/r/{subreddit}/{kind}.json"
        params = {"limit": 100, "raw_json": 1, "t": time_filter}
        if after:
            params["after"] = after
        return self._fetch_listing_payload(url, params=params)

    def _fetch_listing_payload(self, url: str, *, params: dict) -> tuple[List[dict], Optional[str]]:
        payload: dict = {}
        for attempt in range(1, self.listing_retry_max + 1):
            try:
                resp = self._request(
                    url,
                    params=params,
                    timeout=self.request_timeout,
                    failure_key="listing_failures",
                )
                if resp.status_code == 429 and attempt < self.listing_retry_max:
                    self._request_stats["http_429"] += 1
                    time.sleep(1.0 * attempt)
                    continue
                if resp.status_code >= 400:
                    self._request_stats["http_4xx"] += 1
                    return [], None
                payload = resp.json()
                break
            except Exception:
                if attempt >= self.listing_retry_max:
                    return [], None
                time.sleep(0.5 * attempt)
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        children = data.get("children", []) if isinstance(data, dict) else []
        rows: List[dict] = []
        for item in children:
            if isinstance(item, dict) and isinstance(item.get("data"), dict):
                rows.append(item["data"])
        return rows, data.get("after") if isinstance(data, dict) else None

    def _fetch_comments(self, subreddit: str, post_id: str) -> List[str]:
        if self.max_comments_per_post <= 0:
            return []
        url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
        params = {"limit": max(10, self.max_comments_per_post * 4), "depth": 1, "sort": "top", "raw_json": 1}
        payload = None
        for attempt in range(1, self.comments_retry_max + 1):
            try:
                resp = self._request(
                    url,
                    params=params,
                    timeout=self.comments_timeout,
                    failure_key="comment_failures",
                )
                if resp.status_code == 429 and attempt < self.comments_retry_max:
                    self._request_stats["http_429"] += 1
                    time.sleep(0.8)
                    continue
                if resp.status_code >= 400:
                    self._request_stats["http_4xx"] += 1
                    return []
                payload = resp.json()
                break
            except Exception:
                if attempt >= self.comments_retry_max:
                    return []
                time.sleep(0.3)
        if payload is None:
            return []
        if not isinstance(payload, list) or len(payload) < 2:
            return []
        listing = payload[1].get("data", {}).get("children", [])
        comments: List[str] = []
        seen = set()
        for item in listing:
            if not isinstance(item, dict) or item.get("kind") != "t1":
                continue
            data = item.get("data", {})
            body = str(data.get("body") or "").strip()
            if not body or body in {"[deleted]", "[removed]"}:
                continue
            if body in seen:
                continue
            seen.add(body)
            comments.append(body)
            if len(comments) >= self.max_comments_per_post:
                break
        return comments

    def _normalize_post(
        self,
        post: dict,
        *,
        subreddit: str,
        created_utc: int,
        fetch_comments: bool = True,
    ) -> Optional[dict]:
        title = str(post.get("title") or "").strip()
        if not title:
            return None
        post_id = self._normalize_post_id(post)
        permalink = str(post.get("permalink") or "").strip()
        full_link = str(post.get("full_link") or "").strip()
        if full_link:
            url = full_link
        elif permalink:
            if permalink.startswith("http://") or permalink.startswith("https://"):
                url = permalink
            else:
                url = f"https://www.reddit.com{permalink}"
        else:
            url = str(post.get("url") or "")

        timestamp = datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        day = to_day(timestamp)
        if not day:
            return None

        num_comments = self._safe_int(post.get("num_comments")) or 0
        comments = self._fetch_comments(subreddit, post_id) if fetch_comments and post_id and num_comments > 0 else []
        payload = {
            "subreddit": subreddit,
            "score": self._safe_int(post.get("score")) or 0,
            "num_comments": num_comments,
            "title": title,
            "comments": comments,
        }

        rid = stable_record_id("sociomedia/reddit", post_id, day, title)
        return TextRecord(
            id=rid,
            kind="info",
            source="sociomedia/reddit",
            timestamp=day,
            url=url or None,
            payload=payload,
        ).normalized().to_dict()

    @staticmethod
    def _normalize_post_id(post: dict) -> str:
        post_id = str(post.get("id") or "").strip()
        if post_id:
            return post_id
        fullname = str(post.get("name") or "").strip()
        if fullname.startswith("t3_"):
            return fullname[3:]
        return ""

    @staticmethod
    def _safe_int(value) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _request(
        self,
        url: str,
        *,
        params: dict,
        timeout: tuple[float, float],
        failure_key: str,
    ) -> requests.Response:
        try:
            return self.session.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            self._request_stats[failure_key] = self._request_stats.get(failure_key, 0) + 1
            if self._should_retry_direct(exc):
                self._disable_proxy()
                return self.session.get(url, params=params, timeout=timeout)
            raise

    def _should_retry_direct(self, exc: requests.RequestException) -> bool:
        if not self._proxy_enabled or self._proxy_bypassed:
            return False
        if not self.session.proxies:
            return False
        message = str(exc).lower()
        proxy_markers = (
            "proxy",
            "127.0.0.1:7897",
            "failed to establish a new connection",
            "connection refused",
        )
        return any(marker in message for marker in proxy_markers)

    def _disable_proxy(self) -> None:
        if self._proxy_bypassed:
            return
        self.session.proxies.clear()
        self._proxy_bypassed = True
        self._request_stats["proxy_bypass"] = self._request_stats.get("proxy_bypass", 0) + 1
        logger.warning(f"[{self.name}] proxy unavailable; fallback to direct mode")
