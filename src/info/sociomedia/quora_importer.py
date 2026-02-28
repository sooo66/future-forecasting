# -*- coding: utf-8 -*-
"""Quora answer importer."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger

from utils.importer_base import BaseImporter
from utils.models import Record


FIXED_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class QuoraConfig:
    start_date: str
    end_date: str
    output_path: str
    request_timeout: int = 20
    request_retries: int = 2
    request_backoff: float = 1.0
    user_agent: str = FIXED_USER_AGENT


class QuoraImportError(RuntimeError):
    pass


class QuoraAnswerImporter(BaseImporter):
    def __init__(self, config: QuoraConfig) -> None:
        super().__init__(
            timeout=config.request_timeout,
            retries=config.request_retries,
            backoff=config.request_backoff,
            user_agent=config.user_agent,
        )
        self.config = config

    def fetch_answer(self, url: str) -> Optional[Record]:
        try:
            html = self._request_text(url, accept="text/html")
        except RuntimeError as exc:
            raise QuoraImportError(str(exc)) from exc

        answer_json = self._extract_answer_json(html)
        if not answer_json:
            return None

        answer_data = answer_json["data"]["answer"]
        pubtime = self._extract_publish_time(answer_data)
        if not pubtime:
            return None
        if not _in_range(pubtime, self.config.start_date, self.config.end_date):
            return None

        title = self._extract_question_title(answer_data)
        content = self._build_content(answer_data)

        return Record(
            id=str(uuid4()),
            source="sociomedia/quora",
            url=url,
            title=title or "",
            description=None,
            content=content,
            published_at=pubtime,
            pubtime=pubtime,
            language="en",
            tags=None,
        )

    def run(self) -> List[Record]:
        return []

    def _extract_answer_json(self, html_content: str) -> Optional[Dict[str, Any]]:
        pattern = r'push\(("{\\"data\\":{\\"answer\\":.*?}}")\);'
        matches = re.finditer(pattern, html_content, re.DOTALL)
        for match in matches:
            json_str = match.group(1)
            try:
                answer_data = json.loads(json_str)
                answer_data = json.loads(answer_data)
                if (
                    "data" in answer_data
                    and "answer" in answer_data["data"]
                    and "content" in answer_data["data"]["answer"]
                ):
                    return answer_data
            except json.JSONDecodeError:
                continue
        return None

    def _extract_publish_time(self, answer_data: Dict[str, Any]) -> Optional[str]:
        timestamp = answer_data.get("creationTime", 0)
        if not timestamp:
            return None
        seconds = timestamp // 1_000_000
        dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
        return dt.isoformat()

    def _extract_question_title(self, answer_data: Dict[str, Any]) -> str:
        question = answer_data.get("question", {})
        raw_title = question.get("title")
        if raw_title:
            try:
                title_data = json.loads(raw_title)
                sections = title_data.get("sections", [])
                if sections and sections[0].get("spans"):
                    return sections[0]["spans"][0].get("text", "").strip()
            except Exception:
                pass
        return question.get("titlePlaintext", "").strip()

    def _build_content(self, answer_data: Dict[str, Any]) -> str:
        raw_content = answer_data.get("content", {})
        if isinstance(raw_content, str):
            try:
                content_data = json.loads(raw_content)
            except json.JSONDecodeError:
                content_data = {}
        else:
            content_data = raw_content

        parts: List[str] = []
        for section in content_data.get("sections", []):
            if section.get("type") == "image":
                for span in section.get("spans", []):
                    modifiers = span.get("modifiers", {})
                    image_url = modifiers.get("image")
                    if image_url:
                        parts.append(image_url)
                continue

            for span in section.get("spans", []):
                text = span.get("text", "").strip()
                modifiers = span.get("modifiers", {})
                image_url = modifiers.get("image")
                if image_url:
                    parts.append(image_url)
                elif text:
                    parts.append(text)

        return "\n".join(parts).strip()


def _in_range(pubtime_iso: str, start_date: str, end_date: str) -> bool:
    start_dt = BaseImporter._parse_datetime(start_date)
    end_dt = BaseImporter._parse_datetime(end_date)
    pub_dt = BaseImporter._parse_datetime(pubtime_iso)
    if not start_dt or not end_dt or not pub_dt:
        return False
    return start_dt <= pub_dt <= end_dt
