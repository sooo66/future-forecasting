"""代理池加载与环境变量设置"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
        if not self.use_proxy or not self._proxies:
            os.environ.pop("PROXIES", None)
            return []
        os.environ["PROXIES"] = ",".join(self._proxies)
        logger.info(f"代理池加载完成: {len(self._proxies)} 个可用代理")
        return list(self._proxies)

    def _load_proxies(self, proxy_file: Optional[Path]) -> List[str]:
        if not proxy_file:
            logger.info("未指定代理池文件，使用直连模式")
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
