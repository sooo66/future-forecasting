"""代理池加载与环境变量设置"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import quote, urlsplit, urlunsplit

from loguru import logger

_DIRECT_MODE_PROXY_KEYS = (
    "PROXIES",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


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

    def __init__(
        self,
        proxy_file: Optional[Path],
        use_proxy: bool = True,
        *,
        proxy_sample_size: int = 0,
        proxy_min_quality_score: float = 0.0,
        proxy_specs: Optional[List[str]] = None,
    ) -> None:
        self.proxy_file = proxy_file
        self.use_proxy = use_proxy
        self.proxy_sample_size = max(0, int(proxy_sample_size))
        self.proxy_min_quality_score = max(0.0, float(proxy_min_quality_score))
        self._proxy_source = "direct"
        explicit_specs = self._normalize_proxy_specs(proxy_specs)
        if not use_proxy:
            self._proxies = []
            self._proxy_source = "direct"
        elif explicit_specs:
            self._proxies = explicit_specs
            self._proxy_source = "explicit"
        else:
            self._proxies = self._load_proxies(proxy_file)
            self._proxy_source = "file" if self._proxies else "env"

    @property
    def proxies(self) -> List[str]:
        return list(self._proxies)

    def refresh_env(self) -> List[str]:
        if not self.use_proxy:
            removed = []
            for key in _DIRECT_MODE_PROXY_KEYS:
                if key in os.environ:
                    removed.append(key)
                os.environ.pop(key, None)
            if removed:
                logger.debug(f"直连模式已清理代理环境变量: {', '.join(removed)}")
            return []
        if self._proxies:
            active_proxies = self._sample_proxies(self._proxies)
            os.environ["PROXIES"] = ",".join(active_proxies)
            logger.info(
                f"代理加载完成: source={self._proxy_source} total={len(self._proxies)} active={len(active_proxies)} "
                f"sample_size={self.proxy_sample_size or 'all'} min_score={self.proxy_min_quality_score:.2f}"
            )
            return list(active_proxies)

        # 显式指定了代理池文件但未加载到可用代理时，不回退环境代理，避免误用系统代理。
        if self.proxy_file is not None:
            os.environ.pop("PROXIES", None)
            logger.warning("指定 proxy_file 但代理池为空，当前运行回退直连模式")
            return []

        env_proxies = build_crawl4ai_proxy_list()
        if env_proxies:
            os.environ["PROXIES"] = ",".join(env_proxies)
            logger.debug(f"crawl4ai 复用环境代理: {describe_proxy_mode()}")
            return env_proxies

        os.environ.pop("PROXIES", None)
        return []

    @staticmethod
    def _normalize_proxy_specs(proxy_specs: Optional[List[str]]) -> List[str]:
        if not proxy_specs:
            return []
        normalized = [str(item).strip() for item in proxy_specs if str(item).strip()]
        return list(dict.fromkeys(normalized))

    def _sample_proxies(self, proxies: List[str]) -> List[str]:
        candidates = list(dict.fromkeys(str(item).strip() for item in proxies if str(item).strip()))
        if not candidates:
            return []
        random.shuffle(candidates)
        if self.proxy_sample_size <= 0 or self.proxy_sample_size >= len(candidates):
            return candidates
        return random.sample(candidates, self.proxy_sample_size)

    def _extract_proxy_spec_and_score(self, item: Any) -> tuple[Optional[str], Optional[float]]:
        if isinstance(item, str):
            spec = item.strip()
            return (spec or None), None

        if not isinstance(item, dict):
            return None, None

        score: Optional[float] = None
        for key in ("quality_score", "score", "success_rate", "health_score"):
            raw_score = item.get(key)
            if raw_score is None:
                continue
            try:
                score = float(raw_score)
                break
            except (TypeError, ValueError):
                continue

        proxy_spec = str(
            item.get("proxy")
            or item.get("proxy_spec")
            or item.get("spec")
            or item.get("proxy_url")
            or ""
        ).strip()
        if proxy_spec:
            return proxy_spec, score

        host = str(item.get("ip") or item.get("host") or item.get("proxy_ip") or "").strip()
        port = str(item.get("port") or item.get("proxy_port") or "").strip()
        if not host or not port:
            return None, score

        username = str(item.get("user") or item.get("username") or "").strip()
        password = str(item.get("pass") or item.get("password") or "").strip()
        if username and password:
            return f"{host}:{port}:{username}:{password}", score
        return f"{host}:{port}", score

    @staticmethod
    def _expand_raw_proxy_items(raw: Any) -> List[Any]:
        if isinstance(raw, list):
            return list(raw)
        if not isinstance(raw, dict):
            return []
        if isinstance(raw.get("proxies"), list):
            return list(raw.get("proxies") or [])
        raw_items: List[Any] = []
        for value in raw.values():
            if isinstance(value, list):
                raw_items.extend(value)
            elif isinstance(value, dict):
                raw_items.append(value)
        return raw_items

    def _load_proxies(self, proxy_file: Optional[Path]) -> List[str]:
        if not proxy_file:
            logger.debug("未指定代理池文件，使用直连模式")
            return []
        if not proxy_file.exists():
            logger.warning(f"代理池文件不存在: {proxy_file}")
            return []

        raw = json.loads(proxy_file.read_text(encoding="utf-8"))
        raw_items = self._expand_raw_proxy_items(raw)
        proxies: List[str] = []
        skipped_low_score = 0
        skipped_unscored = 0
        for item in raw_items:
            spec, score = self._extract_proxy_spec_and_score(item)
            if not spec:
                continue
            if self.proxy_min_quality_score > 0:
                if score is None:
                    skipped_unscored += 1
                    continue
                if score < self.proxy_min_quality_score:
                    skipped_low_score += 1
                    continue
            proxies.append(spec)
        proxies = list(dict.fromkeys(proxies))
        if proxies:
            logger.info(
                f"代理池读取完成: kept={len(proxies)} "
                f"low_score_skipped={skipped_low_score} unscored_skipped={skipped_unscored}"
            )
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
