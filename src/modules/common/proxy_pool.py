"""代理池加载与环境变量设置"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import quote, urlsplit, urlunsplit

from loguru import logger


@dataclass(frozen=True)
class ProxyEntry:
    host: str
    port: str
    username: Optional[str] = None
    password: Optional[str] = None

    def to_env(self) -> str:
        if self.username and self.password:
            return f"{self.host}:{self.port}:{self.username}:{self.password}"
        return f"{self.host}:{self.port}"


class ProxyManager:
    """读取代理池并输出 Crawl4AI 需要的环境变量格式"""

    def __init__(self, proxy_file: Optional[Path], use_proxy: bool = True) -> None:
        self.proxy_file = proxy_file
        self.use_proxy = use_proxy
        self._proxies = self._load_proxies(proxy_file) if use_proxy else []

    @property
    def proxies(self) -> List[str]:
        return list(self._proxies)

    def refresh_env(self) -> List[str]:
        if not self.use_proxy:
            os.environ.pop("PROXIES", None)
            return []
        if self._proxies:
            os.environ["PROXIES"] = ",".join(self._proxies)
            logger.debug(f"代理池加载完成: {len(self._proxies)} 个可用代理")
            return list(self._proxies)

        env_proxies = build_crawl4ai_proxy_list()
        if env_proxies:
            os.environ["PROXIES"] = ",".join(env_proxies)
            logger.debug(f"crawl4ai 复用环境代理: {describe_proxy_mode()}")
            return env_proxies

        os.environ.pop("PROXIES", None)
        return []

    def _load_proxies(self, proxy_file: Optional[Path]) -> List[str]:
        if not proxy_file:
            logger.debug("未指定代理池文件，使用直连模式")
            return []
        if not proxy_file.exists():
            logger.warning(f"代理池文件不存在: {proxy_file}")
            return []

        raw = json.loads(proxy_file.read_text(encoding="utf-8"))
        entries: List[ProxyEntry] = []

        if isinstance(raw, list):
            raw_items = raw
        elif isinstance(raw, dict):
            raw_items = []
            for value in raw.values():
                if isinstance(value, list):
                    raw_items.extend(value)
        else:
            raw_items = []

        for item in raw_items:
            if not isinstance(item, dict):
                continue
            host = str(item.get("ip") or item.get("host") or item.get("proxy_ip") or "")
            port = str(item.get("port") or item.get("proxy_port") or "")
            if not host or not port:
                continue
            entry = ProxyEntry(
                host=host,
                port=port,
                username=item.get("user") or item.get("username"),
                password=item.get("pass") or item.get("password"),
            )
            entries.append(entry)

        proxies = [entry.to_env() for entry in entries]
        if proxies:
            logger.info(f"代理池读取完成: {len(proxies)} 个")
        else:
            logger.warning("代理池为空，使用直连模式")
        return proxies


def _proxy_env_value(*keys: str) -> str:
    for key in keys:
        value = str(os.getenv(key, "") or "").strip()
        if value:
            return value
    return ""


def _redact_proxy_value(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "://" in raw:
        try:
            parts = urlsplit(raw)
            if parts.username or parts.password:
                host = parts.hostname or ""
                if parts.port:
                    host = f"{host}:{parts.port}"
                netloc = f"***:***@{host}" if host else "***:***"
                return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
        except Exception:
            return raw
        return raw

    parts = raw.split(":")
    if len(parts) >= 4:
        return f"{parts[0]}:{parts[1]}:***:***"
    return raw


def proxy_spec_to_url(proxy_spec: str, *, scheme: str = "http") -> Optional[str]:
    raw = str(proxy_spec or "").strip()
    if not raw:
        return None
    if "://" in raw:
        return raw

    parts = raw.split(":")
    if len(parts) < 2:
        return None

    host = parts[0].strip()
    port = parts[1].strip()
    if not host or not port:
        return None

    auth = ""
    if len(parts) >= 4:
        username = quote(parts[2], safe="")
        password = quote(":".join(parts[3:]), safe="")
        auth = f"{username}:{password}@"

    return f"{scheme}://{auth}{host}:{port}"


def build_requests_proxy_map() -> dict[str, str]:
    """Build requests/httpx-style proxy map from env or PROXIES."""
    https_proxy = _proxy_env_value("HTTPS_PROXY", "https_proxy")
    http_proxy = _proxy_env_value("HTTP_PROXY", "http_proxy")
    all_proxy = _proxy_env_value("ALL_PROXY", "all_proxy")

    if https_proxy or http_proxy or all_proxy:
        return {
            "http": http_proxy or all_proxy or https_proxy,
            "https": https_proxy or http_proxy or all_proxy,
        }

    raw_pool = _proxy_env_value("PROXIES")
    if not raw_pool:
        return {}

    first_spec = next((part.strip() for part in raw_pool.split(",") if part.strip()), "")
    proxy_url = proxy_spec_to_url(first_spec)
    if not proxy_url:
        return {}

    return {"http": proxy_url, "https": proxy_url}


def build_crawl4ai_proxy_list() -> List[str]:
    raw_pool = _proxy_env_value("PROXIES")
    if raw_pool:
        return [part.strip() for part in raw_pool.split(",") if part.strip()]

    https_proxy = _proxy_env_value("HTTPS_PROXY", "https_proxy")
    http_proxy = _proxy_env_value("HTTP_PROXY", "http_proxy")
    all_proxy = _proxy_env_value("ALL_PROXY", "all_proxy")
    selected = https_proxy or http_proxy or all_proxy
    if not selected:
        return []
    return [selected]


def describe_proxy_mode() -> str:
    raw_pool = _proxy_env_value("PROXIES")
    if raw_pool:
        proxies = [part.strip() for part in raw_pool.split(",") if part.strip()]
        if not proxies:
            return "direct"
        sample = _redact_proxy_value(proxies[0])
        extra = "" if len(proxies) == 1 else f" (+{len(proxies) - 1} more)"
        return f"proxy_pool:{sample}{extra}"

    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        value = _proxy_env_value(key)
        if value:
            return f"env:{key}={_redact_proxy_value(value)}"
    return "direct"


def configure_requests_session(session: Any) -> bool:
    """Apply proxy settings to a requests session if any are configured."""
    proxy_map = build_requests_proxy_map()
    session.trust_env = False
    session.proxies.clear()
    if not proxy_map:
        return False
    session.proxies.update(proxy_map)
    logger.info(
        f"requests 会话已配置代理: "
        f"{_redact_proxy_value(proxy_map.get('https') or proxy_map.get('http') or '')}"
    )
    return True
