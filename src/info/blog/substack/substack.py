"""Substack importer for incremental archive fetching."""
from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger
from bs4 import BeautifulSoup

from utils.models import Record
from utils.importer_base import BaseImporter
from info.crawler.extractor import Extractor

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("Python < 3.11 需要安装 tomli: pip install tomli")


DEFAULT_LIMIT = 50
DEFAULT_TIMEOUT = 20
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 1.5


class SubstackImportError(RuntimeError):
    """Raised when Substack import fails in a non-recoverable way."""


@dataclass(frozen=True)
class AuthorConfig:
    """Author configuration for Substack crawling."""
    name: str
    publication: Optional[str] = None
    base_url: Optional[str] = None
    enabled: bool = True
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    domain: Optional[str] = None

    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        if not self.publication:
            raise SubstackImportError(
                f"Author '{self.name}' must set publication or base_url"
            )
        return f"https://{self.publication}.substack.com"

    def state_key(self) -> str:
        if self.domain:
            return f"{self.domain}:{self.name}"
        return self.name


@dataclass
class ImportConfig:
    """Importer configuration loaded from TOML."""
    authors: List[AuthorConfig]
    output_dir: Path
    state_path: Path
    request_timeout: int = DEFAULT_TIMEOUT
    request_retries: int = DEFAULT_RETRIES
    request_backoff: float = DEFAULT_BACKOFF
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )


class SubstackImporter(BaseImporter):
    """Fetch Substack posts incrementally and store them as JSONL."""

    def __init__(self, config: ImportConfig) -> None:
        super().__init__(
            timeout=config.request_timeout,
            retries=config.request_retries,
            backoff=config.request_backoff,
            user_agent=config.user_agent,
        )
        self.config = config
        self.state = self._load_state(config.state_path)
        self._content_cleaner = Extractor("")

    def run(self) -> Dict[str, int]:
        """Process all enabled authors and return counts per author."""
        logger.info(f"Substack import start: {len(self.config.authors)} authors")
        results: Dict[str, int] = {}
        for author in self.config.authors:
            if not author.enabled:
                logger.info(f"Skip disabled author: {author.name}")
                continue
            try:
                count = self.import_author(author)
            except SubstackImportError as exc:
                logger.warning(f"Skip author {author.name} due to error: {exc}")
                continue
            results[author.name] = count
        self._save_state(self.config.state_path, self.state)
        logger.info("Substack import done")
        return results

    def import_author(self, author: AuthorConfig) -> int:
        base_url = author.resolved_base_url()
        last_seen = self._get_last_seen(author)
        start_date = self._parse_datetime(author.start_date)
        end_date = self._parse_datetime(author.end_date)
        cutoff = max(filter(None, [last_seen, start_date]), default=None)

        output_path = self._output_path(author)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Import author={author.name} base={base_url} start={author.start_date} "
            f"end={author.end_date} last_seen={last_seen.isoformat() if last_seen else None} "
            f"output={output_path}"
        )
        new_posts = self._fetch_new_posts(base_url, author, cutoff, end_date)
        if not new_posts:
            logger.info(f"No new posts for {author.name}")
            return 0

        with output_path.open("a", encoding="utf-8") as f:
            for post in new_posts:
                record = self._record_from_post(post, author, base_url)
                if not record:
                    continue
                f.write(record.to_json())
                f.write("\n")

        parsed_dates = [self._parse_datetime(p.get("post_date")) for p in new_posts]
        parsed_dates = [dt for dt in parsed_dates if dt]
        latest = max(parsed_dates, default=cutoff)
        if latest:
            self.state[author.state_key()] = {"last_post_date": latest.isoformat()}
        return len(new_posts)

    def _fetch_new_posts(
        self,
        base_url: str,
        author: AuthorConfig,
        cutoff: Optional[datetime],
        end_date: Optional[datetime],
    ) -> List[Dict[str, Any]]:
        all_posts: List[Dict[str, Any]] = []
        offset = 0
        logger.info(f"Fetch archive: {base_url}")
        while True:
            logger.info(f"Archive page offset={offset} limit={DEFAULT_LIMIT}")
            posts = self._fetch_archive_page(
                base_url, offset=offset, limit=DEFAULT_LIMIT, sort="new", audience="everyone"
            )
            if not posts:
                break

            for post in posts:
                normalized = self._normalize_post(post, author, base_url)
                post_dt = self._parse_datetime(normalized.get("post_date"))
                if end_date and post_dt and post_dt > end_date:
                    continue
                if cutoff and post_dt and post_dt <= cutoff:
                    return all_posts
                if not self._is_free_post(post):
                    continue
                slug = normalized.get("slug")
                if slug:
                    detail = self._fetch_post_detail(base_url, slug)
                    normalized = self._merge_post_detail(normalized, detail)
                normalized["content"] = self._clean_post_content(
                    normalized.get("body_html"),
                    normalized.get("body"),
                    normalized.get("description"),
                )
                all_posts.append(normalized)

            offset += len(posts)
            logger.info(f"Fetched {len(all_posts)} posts (offset={offset})")
        return all_posts

    def _normalize_post(
        self,
        post: Dict[str, Any],
        author: AuthorConfig,
        base_url: str,
    ) -> Dict[str, Any]:
        post_id = post.get("id")
        slug = post.get("slug")
        canonical_url = post.get("canonical_url")
        if not canonical_url and slug:
            canonical_url = f"{base_url}/p/{slug}"
        return {
            "id": post_id,
            "title": post.get("title") or post.get("headline"),
            "description": post.get("description") or post.get("subtitle"),
            "slug": slug,
            "post_date": post.get("post_date"),
            "canonical_url": canonical_url,
            "audience": post.get("audience"),
            "author": author.name,
            "domain": author.domain,
        }

    def _get_last_seen(self, author: AuthorConfig) -> Optional[datetime]:
        entry = self.state.get(author.state_key())
        if not entry:
            return None
        return self._parse_datetime(entry.get("last_post_date"))

    def _output_path(self, author: AuthorConfig) -> Path:
        return self.config.output_dir / "substack_all.jsonl"

    def _is_free_post(self, post: Dict[str, Any]) -> bool:
        audience = (post.get("audience") or "").lower()
        if audience in {"paid", "only_paid"}:
            return False
        if post.get("is_paid") is True:
            return False
        if post.get("paywall") or post.get("paywalled"):
            return False
        return True

    def _merge_post_detail(
        self, base: Dict[str, Any], detail: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not detail:
            return base
        merged = dict(base)
        for key in (
            "title",
            "description",
            "subtitle",
            "post_date",
            "canonical_url",
            "slug",
            "body",
            "body_html",
        ):
            value = detail.get(key)
            if value not in (None, ""):
                merged[key] = value
        return merged

    def _clean_post_content(
        self,
        body_html: Optional[str],
        body: Optional[str],
        fallback: Optional[str],
    ) -> str:
        if body:
            return self._content_cleaner.clean_fit_markdown(body)
        if body_html:
            text = self._html_to_text(body_html)
            return self._content_cleaner.clean_fit_markdown(text)
        return fallback or ""

    def _html_to_text(self, html: str) -> str:
        soup = BeautifulSoup(html or "", "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text("\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    def _record_from_post(
        self,
        post: Dict[str, Any],
        author: AuthorConfig,
        base_url: str,
    ) -> Optional[Record]:
        post_date = post.get("post_date")
        post_dt = self._parse_datetime(post_date)
        if not post_dt:
            logger.warning(f"Skip post without valid post_date: {post.get('id')}")
            return None
        url = post.get("canonical_url")
        if not url:
            logger.warning(f"Skip post without URL: {post.get('id')}")
            return None

        content = post.get("content") or ""
        if not content:
            logger.warning(f"Empty content for post: {post.get('id')}")

        tags = self._extract_tags(post, author)
        pubtime = post_dt.isoformat()
        return Record(
            id=str(uuid4()),
            source=f"substack/{author.publication or author.name}",
            url=url,
            title=post.get("title") or "",
            description=post.get("description"),
            content=content,
            published_at=pubtime,
            pubtime=pubtime,
            language="en",
            tags=tags or None,
        )

    def _extract_tags(
        self, post: Dict[str, Any], author: AuthorConfig
    ) -> List[str]:
        tags: List[str] = []
        if author.domain:
            tags.append(author.domain)
        topics = post.get("topics")
        if isinstance(topics, list):
            for item in topics:
                if isinstance(item, dict) and item.get("name"):
                    tags.append(item["name"])
                elif isinstance(item, str):
                    tags.append(item)
        elif isinstance(topics, dict) and topics.get("name"):
            tags.append(topics["name"])
        return tags

    def _fetch_archive_page(
        self,
        base_url: str,
        *,
        offset: int,
        limit: int = DEFAULT_LIMIT,
        sort: str = "new",
        audience: str = "everyone",
    ) -> List[Dict[str, Any]]:
        """Fetch archive posts."""
        url = f"{base_url}/api/v1/archive"
        params = {"sort": sort, "offset": offset, "limit": limit, "audience": audience}
        try:
            payload = self._request_json(url, params=params)
        except RuntimeError as exc:
            raise SubstackImportError(str(exc)) from exc
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            posts = payload.get("posts") or payload.get("items")
            if isinstance(posts, list):
                return posts
        logger.warning(f"Unexpected archive payload format from {url}")
        return []

    def _fetch_post_detail(self, base_url: str, slug: str) -> Dict[str, Any]:
        url = f"{base_url}/api/v1/posts/{slug}"
        try:
            payload = self._request_json(url)
        except RuntimeError as exc:
            raise SubstackImportError(str(exc)) from exc
        return payload if isinstance(payload, dict) else {}


def load_import_config(config_path: str | Path) -> ImportConfig:
    """Load importer config from a TOML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Author config not found: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SubstackImportError(f"Invalid author config: {path}") from exc

    authors: List[AuthorConfig] = []
    domains = raw.get("domains", {})
    if not domains:
        raise SubstackImportError("Config must define [domains.*] entries")

    date_range = raw.get("date_range", {})
    default_start = date_range.get("start_date")
    default_end = date_range.get("end_date")

    for domain_name, domain in domains.items():
        for entry in domain.get("authors", []):
            if not entry.get("name"):
                raise SubstackImportError("Author entry missing name")
            authors.append(
                AuthorConfig(
                    name=entry["name"],
                    publication=entry.get("publication"),
                    base_url=entry.get("base_url"),
                    enabled=entry.get("enabled", True),
                    start_date=entry.get("start_date") or default_start,
                    end_date=entry.get("end_date") or default_end,
                    domain=domain_name,
                )
            )

    paths = raw.get("paths", {})
    output_dir = Path(paths.get("output_dir", "data/processed/substack"))
    state_path = Path(paths.get("state_path", "data/processed/substack_state.json"))
    request = raw.get("request", {})
    return ImportConfig(
        authors=authors,
        output_dir=output_dir,
        state_path=state_path,
        request_timeout=request.get("timeout", DEFAULT_TIMEOUT),
        request_retries=request.get("retries", DEFAULT_RETRIES),
        request_backoff=request.get("backoff", DEFAULT_BACKOFF),
        user_agent=request.get(
            "user_agent",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36",
        ),
    )


def run_from_config(config_path: str | Path) -> Dict[str, int]:
    """Convenience entrypoint to run importer from a config file."""
    config = load_import_config(config_path)
    importer = SubstackImporter(config)
    return importer.run()


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/substack_authors.toml")
    results = run_from_config(config_path)
    for author, count in results.items():
        print(f"{author}: {count}")
