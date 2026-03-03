"""Reddit public JSON harvester (read-only, no OAuth)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_SUBREDDITS = [
    "worldnews",
    "geopolitics",
    "economics",
    "finance",
    "energy",
    "climate",
    "science",
    "technology",
    "futurology",
    "supplychain",
]

DEFAULT_USER_AGENT = "forecastkb-reddit-harvester/0.1 (contact: youremail@example.com)"


@dataclass
class OutputPaths:
    out_root: Path
    submissions_jsonl: Path
    state_json: Path
    seen_sqlite: Path
    harvest_log: Path
    coverage_json: Path
    failures_jsonl: Path


def build_output_paths(out_dir: str | Path) -> OutputPaths:
    out = Path(out_dir)
    return OutputPaths(
        out_root=out,
        submissions_jsonl=out / "submissions" / "submissions.jsonl",
        state_json=out / "state" / "state.json",
        seen_sqlite=out / "state" / "seen_ids.sqlite",
        harvest_log=out / "logs" / "harvest.log",
        coverage_json=out / "reports" / "coverage.json",
        failures_jsonl=out / "failures" / "failures.jsonl",
    )


def utc_now_ts() -> int:
    return int(time.time())


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_date_str(utc_ts: int) -> str:
    return datetime.fromtimestamp(utc_ts, tz=timezone.utc).strftime("%Y-%m-%d")


def setup_logging(log_path: Path, verbose: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO if not verbose else logging.DEBUG)
    logger.addHandler(stream_handler)


def parse_subreddits(value: str | None) -> list[str]:
    if not value:
        return DEFAULT_SUBREDDITS[:]
    parts = [x.strip() for x in value.split(",")]
    cleaned = [x for x in parts if x]
    if not cleaned:
        return DEFAULT_SUBREDDITS[:]
    return cleaned


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "backfill": {"per_subreddit": {}},
            "follow": {"per_subreddit_last_seen_utc": {}},
            "last_run_started_at": None,
            "last_run_finished_at": None,
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logging.exception("Failed to load state from %s; using empty defaults", path)
        return {
            "backfill": {"per_subreddit": {}},
            "follow": {"per_subreddit_last_seen_utc": {}},
            "last_run_started_at": None,
            "last_run_finished_at": None,
        }


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class SeenStore:
    def __init__(self, sqlite_path: Path) -> None:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(sqlite_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seen (
              post_id TEXT PRIMARY KEY,
              subreddit TEXT,
              created_utc INTEGER,
              seen_at INTEGER
            )
            """
        )
        self.conn.commit()

    def add_if_new(self, post_id: str, subreddit: str, created_utc: int) -> bool:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO seen(post_id, subreddit, created_utc, seen_at) VALUES (?, ?, ?, ?)",
            (post_id, subreddit, int(created_utc), utc_now_ts()),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def close(self) -> None:
        self.conn.close()


class RateLimiter:
    def __init__(self, base_rate: float, min_subreddit_interval_sec: float = 30.0) -> None:
        self.base_rate = max(0.05, float(base_rate))
        self.adaptive_rate = self.base_rate
        self.min_subreddit_interval_sec = min_subreddit_interval_sec
        self._last_global_ts = 0.0
        self._last_subreddit_ts: dict[str, float] = {}

    def wait(self, subreddit: str) -> None:
        now = time.time()
        global_interval = 1.0 / max(0.05, min(self.base_rate, self.adaptive_rate))
        next_global = self._last_global_ts + global_interval
        next_sub = self._last_subreddit_ts.get(subreddit, 0.0) + self.min_subreddit_interval_sec
        sleep_for = max(0.0, next_global - now, next_sub - now)
        if sleep_for > 0:
            time.sleep(sleep_for)
        mark = time.time()
        self._last_global_ts = mark
        self._last_subreddit_ts[subreddit] = mark

    def update_from_headers(self, headers: requests.structures.CaseInsensitiveDict[str]) -> None:
        remain_raw = headers.get("X-Ratelimit-Remaining")
        reset_raw = headers.get("X-Ratelimit-Reset")
        if remain_raw is None or reset_raw is None:
            self.adaptive_rate = min(self.base_rate, self.adaptive_rate * 1.05)
            return

        try:
            remaining = float(remain_raw)
            reset = float(reset_raw)
        except ValueError:
            return
        if reset <= 0:
            return

        recommended = max(0.05, (remaining - 1.0) / max(1.0, reset))
        if remaining <= 5 or recommended < self.adaptive_rate:
            self.adaptive_rate = max(0.05, min(self.base_rate, recommended))
        else:
            self.adaptive_rate = min(self.base_rate, self.adaptive_rate * 1.1)


class RedditPublicJsonHarvester:
    def __init__(
        self,
        *,
        out_paths: OutputPaths,
        subreddits: list[str],
        rate: float,
        user_agent: str,
        max_retries: int = 6,
    ) -> None:
        self.out_paths = out_paths
        self.subreddits = subreddits
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(base_rate=rate, min_subreddit_interval_sec=30.0)
        self.state = load_state(out_paths.state_json)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "application/json",
            }
        )
        self.timeout = (10, 60)
        self.seen_store = SeenStore(out_paths.seen_sqlite)

    def close(self) -> None:
        self.seen_store.close()
        self.session.close()

    def _append_jsonl(self, path: Path, obj: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")

    def _record_failure(
        self,
        *,
        subreddit: str,
        url: str,
        params: dict[str, Any] | None,
        status_code: int | None,
        error: str,
        attempt: int,
    ) -> None:
        failure = {
            "timestamp_utc": iso_now(),
            "subreddit": subreddit,
            "url": url,
            "params": params or {},
            "status_code": status_code,
            "error": error,
            "attempt": attempt,
        }
        self._append_jsonl(self.out_paths.failures_jsonl, failure)

    def _request_with_retry(
        self, subreddit: str, url: str, params: dict[str, Any] | None = None
    ) -> requests.Response | None:
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                sleep_sec = min(60, 2 ** (attempt - 1)) + random.uniform(0, 0.25)
                time.sleep(sleep_sec)

            self.rate_limiter.wait(subreddit)
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    self._record_failure(
                        subreddit=subreddit,
                        url=url,
                        params=params,
                        status_code=None,
                        error=f"request_exception: {exc}",
                        attempt=attempt,
                    )
                    logging.error("[%s] request exception after retries: %s", subreddit, exc)
                    return None
                continue

            self.rate_limiter.update_from_headers(resp.headers)
            if resp.status_code == 200:
                return resp

            if resp.status_code in (429, 503, 504):
                if attempt >= self.max_retries:
                    self._record_failure(
                        subreddit=subreddit,
                        url=url,
                        params=params,
                        status_code=resp.status_code,
                        error=f"http_status_{resp.status_code}",
                        attempt=attempt,
                    )
                    logging.error("[%s] http %s after retries", subreddit, resp.status_code)
                    return None

                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        retry_after_sec = min(60, max(0, int(float(retry_after))))
                        time.sleep(retry_after_sec)
                    except ValueError:
                        pass
                continue

            self._record_failure(
                subreddit=subreddit,
                url=url,
                params=params,
                status_code=resp.status_code,
                error=f"non_retriable_http_{resp.status_code}",
                attempt=attempt,
            )
            logging.error("[%s] non-retriable http status: %s", subreddit, resp.status_code)
            return None
        return None

    def _fetch_new_listing(
        self, subreddit: str, after: str | None = None
    ) -> tuple[list[dict[str, Any]], str | None]:
        url = f"https://www.reddit.com/r/{subreddit}/new.json"
        params: dict[str, Any] = {"limit": 100}
        if after:
            params["after"] = after
        resp = self._request_with_retry(subreddit, url, params)
        if resp is None:
            return [], None
        try:
            payload = resp.json()
        except ValueError:
            self._record_failure(
                subreddit=subreddit,
                url=url,
                params=params,
                status_code=200,
                error="invalid_json",
                attempt=0,
            )
            return [], None

        data = payload.get("data", {})
        children = data.get("children") or []
        next_after = data.get("after")
        rows: list[dict[str, Any]] = []
        for child in children:
            if not isinstance(child, dict):
                continue
            row = child.get("data")
            if isinstance(row, dict):
                rows.append(row)
        return rows, next_after

    def _fetch_comments(self, subreddit: str, post_id: str, max_comments: int) -> list[str]:
        url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
        # Request a slightly larger pool, then filter AutoModerator/stickied/invalid comments.
        request_limit = max(10, max_comments * 4)
        params = {"limit": request_limit, "depth": 1, "sort": "top", "raw_json": 1}
        resp = self._request_with_retry(subreddit, url, params)
        if resp is None:
            return []
        try:
            payload = resp.json()
        except ValueError:
            return []

        if not isinstance(payload, list) or len(payload) < 2:
            return []
        comments_listing = payload[1].get("data", {}).get("children", [])
        comments: list[str] = []
        seen_bodies: set[str] = set()
        for item in comments_listing:
            if not isinstance(item, dict):
                continue
            if item.get("kind") != "t1":
                continue
            data = item.get("data", {}) if isinstance(item, dict) else {}
            body = data.get("body")
            if not isinstance(body, str):
                continue
            body = body.strip()
            if not body or body in ("[deleted]", "[removed]"):
                continue

            author = str(data.get("author") or "")
            distinguished = data.get("distinguished")
            is_stickied = bool(data.get("stickied"))
            lower = body.lower()
            is_automod_template = (
                author.lower() == "automoderator"
                or ("i am a bot" in lower and "contact the moderators" in lower)
                or (is_stickied and distinguished == "moderator")
            )
            if is_automod_template:
                continue
            if body in seen_bodies:
                continue
            seen_bodies.add(body)
            comments.append(body)
            if len(comments) >= max_comments:
                break
        return comments

    def _to_record(
        self,
        post: dict[str, Any],
        *,
        subreddit: str,
        endpoint: str,
        include_comments: bool,
        max_comments_per_post: int,
    ) -> dict[str, Any] | None:
        post_id = post.get("id")
        created_utc = post.get("created_utc")
        if not post_id or created_utc is None:
            return None
        try:
            created_utc_int = int(float(created_utc))
        except (TypeError, ValueError):
            return None

        if not self.seen_store.add_if_new(str(post_id), subreddit, created_utc_int):
            return None

        comments: list[str] = []
        if include_comments and max_comments_per_post > 0:
            comments = self._fetch_comments(subreddit, str(post_id), max_comments_per_post)

        permalink = post.get("permalink")
        post_url = (
            f"https://www.reddit.com{permalink}"
            if isinstance(permalink, str) and permalink
            else post.get("url")
        )
        if not isinstance(post_url, str):
            post_url = ""

        return {
            "id": str(uuid.uuid4()),
            "source": "sociomedia/reddit",
            "subreddit": subreddit,
            "post_id": str(post_id),
            "created_utc": created_utc_int,
            "timestamp": utc_date_str(created_utc_int),
            "title": str(post.get("title") or ""),
            "text": str(post.get("selftext") or ""),
            "score": int(post.get("score") or 0),
            "num_comments": int(post.get("num_comments") or 0),
            "url": post_url,
            "endpoint": endpoint,
            "comment": comments,
        }

    def _write_submission(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.out_paths.submissions_jsonl, record)

    def run_backfill(
        self,
        *,
        days: int,
        max_pages: int,
        max_posts_per_subreddit: int,
        stuck_k: int,
        include_comments: bool,
        max_comments_per_post: int,
    ) -> None:
        now_utc = utc_now_ts()
        cutoff_utc = now_utc - max(1, days) * 86400
        self.state["last_run_started_at"] = iso_now()
        self.state.setdefault("backfill", {}).setdefault("per_subreddit", {})
        coverage: dict[str, Any] = {}

        for subreddit in self.subreddits:
            sub_state = self.state["backfill"]["per_subreddit"].setdefault(
                subreddit,
                {"after": None, "oldest_seen_utc": None, "done": False},
            )
            if sub_state.get("done"):
                logging.info("[%s] skipped; already marked done in state", subreddit)
                coverage[subreddit] = {
                    "reason": "already_done",
                    "pages_fetched": 0,
                    "posts_written": 0,
                    "oldest_seen_utc": sub_state.get("oldest_seen_utc"),
                }
                continue

            after = sub_state.get("after")
            pages = 0
            posts_written = 0
            reason = "finished"
            prev_after: str | None = None
            prev_oldest: int | None = None
            stuck_count = 0

            while pages < max_pages and posts_written < max_posts_per_subreddit:
                rows, next_after = self._fetch_new_listing(subreddit, after)
                pages += 1
                if not rows:
                    reason = "request_failed_or_empty"
                    sub_state["done"] = True
                    break

                min_created_in_page: int | None = None
                for row in rows:
                    created_raw = row.get("created_utc")
                    try:
                        created_utc = int(float(created_raw))
                    except (TypeError, ValueError):
                        continue

                    if min_created_in_page is None or created_utc < min_created_in_page:
                        min_created_in_page = created_utc

                    if created_utc < cutoff_utc:
                        continue

                    rec = self._to_record(
                        row,
                        subreddit=subreddit,
                        endpoint="new",
                        include_comments=include_comments,
                        max_comments_per_post=max_comments_per_post,
                    )
                    if rec is None:
                        continue
                    self._write_submission(rec)
                    posts_written += 1
                    if posts_written >= max_posts_per_subreddit:
                        break

                if min_created_in_page is not None:
                    sub_state["oldest_seen_utc"] = min_created_in_page

                if (
                    prev_oldest is not None
                    and min_created_in_page is not None
                    and min_created_in_page >= prev_oldest
                ):
                    stuck_count += 1
                elif prev_after is not None and next_after == prev_after:
                    stuck_count += 1
                else:
                    stuck_count = 0

                prev_oldest = min_created_in_page
                prev_after = after

                if min_created_in_page is not None and min_created_in_page < cutoff_utc:
                    reason = "reached_cutoff"
                    sub_state["done"] = True
                    sub_state["after"] = next_after
                    break

                if not next_after:
                    reason = "no_after"
                    sub_state["done"] = True
                    sub_state["after"] = None
                    break

                if stuck_count >= stuck_k:
                    reason = "listing_limit"
                    sub_state["done"] = True
                    sub_state["after"] = next_after
                    break

                after = next_after
                sub_state["after"] = after
                save_state(self.out_paths.state_json, self.state)

            if pages >= max_pages and not sub_state.get("done"):
                reason = "max_pages"
            if posts_written >= max_posts_per_subreddit and not sub_state.get("done"):
                reason = "max_posts_per_subreddit"

            coverage[subreddit] = {
                "reason": reason,
                "pages_fetched": pages,
                "posts_written": posts_written,
                "oldest_seen_utc": sub_state.get("oldest_seen_utc"),
                "cutoff_utc": cutoff_utc,
                "done": bool(sub_state.get("done")),
            }
            logging.info(
                "[%s] backfill done. reason=%s pages=%s posts_written=%s oldest_seen_utc=%s",
                subreddit,
                reason,
                pages,
                posts_written,
                sub_state.get("oldest_seen_utc"),
            )
            save_state(self.out_paths.state_json, self.state)

        coverage_payload = {"generated_at": iso_now(), "mode": "backfill", "per_subreddit": coverage}
        self.out_paths.coverage_json.parent.mkdir(parents=True, exist_ok=True)
        self.out_paths.coverage_json.write_text(
            json.dumps(coverage_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.state["last_run_finished_at"] = iso_now()
        save_state(self.out_paths.state_json, self.state)

    def run_follow(
        self,
        *,
        poll_interval_sec: int,
        include_comments: bool,
        max_comments_per_post: int,
    ) -> None:
        self.state["last_run_started_at"] = iso_now()
        self.state.setdefault("follow", {}).setdefault("per_subreddit_last_seen_utc", {})
        save_state(self.out_paths.state_json, self.state)

        logging.info("Starting follow loop. press Ctrl+C to stop.")
        try:
            while True:
                loop_start = time.time()
                for subreddit in self.subreddits:
                    rows, _ = self._fetch_new_listing(subreddit, after=None)
                    if not rows:
                        continue
                    last_seen = int(
                        self.state["follow"]["per_subreddit_last_seen_utc"].get(subreddit, 0) or 0
                    )
                    max_seen = last_seen
                    written = 0

                    for row in rows:
                        created_raw = row.get("created_utc")
                        try:
                            created_utc = int(float(created_raw))
                        except (TypeError, ValueError):
                            continue
                        if created_utc <= last_seen:
                            continue

                        rec = self._to_record(
                            row,
                            subreddit=subreddit,
                            endpoint="new",
                            include_comments=include_comments,
                            max_comments_per_post=max_comments_per_post,
                        )
                        if rec is None:
                            continue
                        self._write_submission(rec)
                        written += 1
                        if created_utc > max_seen:
                            max_seen = created_utc

                    if max_seen > last_seen:
                        self.state["follow"]["per_subreddit_last_seen_utc"][subreddit] = max_seen
                        save_state(self.out_paths.state_json, self.state)
                    logging.info(
                        "[%s] follow polled=%s written=%s last_seen_utc=%s",
                        subreddit,
                        len(rows),
                        written,
                        self.state["follow"]["per_subreddit_last_seen_utc"].get(subreddit, last_seen),
                    )

                elapsed = time.time() - loop_start
                sleep_for = max(0.0, float(poll_interval_sec) - elapsed)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except KeyboardInterrupt:
            logging.info("Follow stopped by user")
        finally:
            self.state["last_run_finished_at"] = iso_now()
            save_state(self.out_paths.state_json, self.state)


def command_stats(out_paths: OutputPaths) -> None:
    path = out_paths.submissions_jsonl
    if not path.exists():
        print(json.dumps({"submissions_file": str(path), "exists": False}, ensure_ascii=False, indent=2))
        return

    total = 0
    invalid = 0
    per_subreddit: dict[str, dict[str, Any]] = {}
    min_utc: int | None = None
    max_utc: int | None = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue

            total += 1
            sub = str(row.get("subreddit") or "unknown")
            created_utc = row.get("created_utc")
            try:
                created_utc_int = int(created_utc)
            except (TypeError, ValueError):
                created_utc_int = None

            item = per_subreddit.setdefault(
                sub,
                {"count": 0, "min_created_utc": None, "max_created_utc": None},
            )
            item["count"] += 1
            if created_utc_int is not None:
                if item["min_created_utc"] is None or created_utc_int < item["min_created_utc"]:
                    item["min_created_utc"] = created_utc_int
                if item["max_created_utc"] is None or created_utc_int > item["max_created_utc"]:
                    item["max_created_utc"] = created_utc_int
                if min_utc is None or created_utc_int < min_utc:
                    min_utc = created_utc_int
                if max_utc is None or created_utc_int > max_utc:
                    max_utc = created_utc_int

    result = {
        "submissions_file": str(path),
        "exists": True,
        "total_records": total,
        "invalid_json_lines": invalid,
        "global_min_created_utc": min_utc,
        "global_max_created_utc": max_utc,
        "global_min_date": utc_date_str(min_utc) if min_utc is not None else None,
        "global_max_date": utc_date_str(max_utc) if max_utc is not None else None,
        "per_subreddit": per_subreddit,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reddit Public JSON Harvester (read-only, no OAuth)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--subreddits",
        type=str,
        default=",".join(DEFAULT_SUBREDDITS),
        help="Comma-separated subreddits",
    )
    common.add_argument("--out", type=str, default="./data/reddit", help="Output root directory")
    common.add_argument("--rate", type=float, default=1.0, help="Global request rate (req/sec)")
    common.add_argument("--user-agent", type=str, default=DEFAULT_USER_AGENT, help="HTTP User-Agent")
    common.add_argument("--verbose", action="store_true", help="Enable debug logging")
    common.add_argument(
        "--fetch-comments",
        action="store_true",
        help="Fetch a small number of top-level comments per post",
    )
    common.add_argument(
        "--max-comments-per-post",
        type=int,
        default=3,
        help="Max comments per post when --fetch-comments is enabled",
    )

    backfill = subparsers.add_parser("backfill", parents=[common], help="Backfill historical posts")
    backfill.add_argument("--days", type=int, default=90, help="Look-back days")
    backfill.add_argument("--max-pages", type=int, default=200, help="Max pages per subreddit")
    backfill.add_argument(
        "--max-posts-per-subreddit",
        type=int,
        default=5000,
        help="Max posts written per subreddit",
    )
    backfill.add_argument(
        "--stuck-k",
        type=int,
        default=3,
        help="Stuck threshold for listing-limit detection",
    )

    follow = subparsers.add_parser("follow", parents=[common], help="Incremental follow mode")
    follow.add_argument(
        "--poll-interval-sec",
        type=int,
        default=60,
        help="Polling interval in seconds",
    )

    stats = subparsers.add_parser("stats", help="Show local stats")
    stats.add_argument("--out", type=str, default="./data/reddit", help="Output root directory")

    return parser


def run_backfill(args: argparse.Namespace) -> None:
    paths = build_output_paths(args.out)
    setup_logging(paths.harvest_log, verbose=args.verbose)
    subs = parse_subreddits(args.subreddits)
    logging.info("Backfill started: subreddits=%s days=%s", subs, args.days)
    harvester = RedditPublicJsonHarvester(
        out_paths=paths,
        subreddits=subs,
        rate=args.rate,
        user_agent=args.user_agent,
    )
    try:
        harvester.run_backfill(
            days=args.days,
            max_pages=args.max_pages,
            max_posts_per_subreddit=args.max_posts_per_subreddit,
            stuck_k=args.stuck_k,
            include_comments=bool(args.fetch_comments),
            max_comments_per_post=max(0, int(args.max_comments_per_post)),
        )
    finally:
        harvester.close()
    logging.info("Backfill finished.")


def run_follow(args: argparse.Namespace) -> None:
    paths = build_output_paths(args.out)
    setup_logging(paths.harvest_log, verbose=args.verbose)
    subs = parse_subreddits(args.subreddits)
    logging.info(
        "Follow started: subreddits=%s poll_interval_sec=%s",
        subs,
        args.poll_interval_sec,
    )
    harvester = RedditPublicJsonHarvester(
        out_paths=paths,
        subreddits=subs,
        rate=args.rate,
        user_agent=args.user_agent,
    )
    try:
        harvester.run_follow(
            poll_interval_sec=max(1, int(args.poll_interval_sec)),
            include_comments=bool(args.fetch_comments),
            max_comments_per_post=max(0, int(args.max_comments_per_post)),
        )
    finally:
        harvester.close()
    logging.info("Follow finished.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "backfill":
        run_backfill(args)
    elif args.command == "follow":
        run_follow(args)
    elif args.command == "stats":
        command_stats(build_output_paths(args.out))
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
