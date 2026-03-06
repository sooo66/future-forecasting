"""Shared importer utilities and base class."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import requests
from dateutil import parser as date_parser
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from modules.common.proxy_pool import configure_requests_session


class BaseImporter(ABC):
    """Base importer with shared HTTP/session and helpers."""

    def __init__(
        self,
        *,
        timeout: int,
        retries: int,
        backoff: float,
        user_agent: str,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.user_agent = user_agent
        self.session = self._build_session()
        self._proxy_enabled = configure_requests_session(self.session)
        self._proxy_bypassed = False

    @abstractmethod
    def run(self) -> Any:
        """Execute import and return summary or records."""

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=self.retries,
            backoff_factor=self.backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            }
        )
        return session

    def _request_json(self, url: str, params: Optional[dict] = None) -> Any:
        try:
            response = self._request("GET", url, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error(f"Request failed: {url} ({exc})")
            raise RuntimeError(f"Request failed: {url}") from exc
        try:
            return response.json()
        except ValueError as exc:
            logger.error(f"Invalid JSON response: {url}")
            raise RuntimeError(f"Invalid JSON response: {url}") from exc

    def _request_text(
        self, url: str, *, params: Optional[dict] = None, accept: str = "text/plain"
    ) -> str:
        headers = {"Accept": accept, "User-Agent": self.user_agent}
        try:
            response = self._request("GET", url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error(f"Request failed: {url} ({exc})")
            raise RuntimeError(f"Request failed: {url}") from exc
        return response.text

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        try:
            return self.session.request(method, url, **kwargs)
        except requests.RequestException as exc:
            if self._should_retry_direct(exc):
                self._disable_proxy()
                return self.session.request(method, url, **kwargs)
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
        logger.warning("检测到代理不可用，当前 importer 会话已回退到直连模式")

    @staticmethod
    def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            dt = date_parser.isoparse(value)
        except (ValueError, TypeError) as exc:
            logger.warning(f"Invalid datetime value: {value} ({exc})")
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _load_state(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(f"Failed to read state file: {path}")
            raise RuntimeError(f"Invalid state file: {path}") from exc

    @staticmethod
    def _save_state(path: Path, state: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _write_jsonl(records: Iterable[Any], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                if hasattr(record, "to_json"):
                    f.write(record.to_json())
                else:
                    f.write(json.dumps(asdict(record), ensure_ascii=False))
                f.write("\n")
