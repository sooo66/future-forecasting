"""Minimal OpenAI-compatible client wrapper for forecasting experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional

import httpx
from openai import OpenAI

from utils.env import get_first_env, load_dotenv


@dataclass
class LLMUsage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    def to_dict(self) -> dict[str, Optional[int]]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def plus(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            prompt_tokens=_sum_optional_ints(self.prompt_tokens, other.prompt_tokens),
            completion_tokens=_sum_optional_ints(self.completion_tokens, other.completion_tokens),
            total_tokens=_sum_optional_ints(self.total_tokens, other.total_tokens),
        )


class OpenAIChatModel:
    def __init__(self, project_root: Path) -> None:
        load_dotenv(project_root / ".env", override=False)
        self.model = get_first_env("MODEL_NAME", "QWEN_MODEL", "LLM_MODEL", "OPENAI_MODEL")
        self.base_url = get_first_env(
            "V_API_BASE_URL",
            "QWEN_MODEL_SERVER",
            "OPENAI_BASE_URL",
            "BASE_URL",
            "LLM_BASE_URL",
            "MODEL_SERVER",
        )
        self.api_key = get_first_env(
            "V_API_KEY",
            "QWEN_API_KEY",
            "DASHSCOPE_API_KEY",
            "OPENAI_API_KEY",
            "API_KEY",
            "LLM_API_KEY",
        )
        if not self.model:
            raise ValueError("MODEL_NAME / QWEN_MODEL / LLM_MODEL / OPENAI_MODEL is required")
        self.client = OpenAI(
            base_url=self.base_url or None,
            api_key=self.api_key or None,
            timeout=120.0,
            max_retries=2,
            http_client=httpx.Client(trust_env=False),
        )

    def to_agent_config(self) -> dict[str, str]:
        config = {"model": self.model}
        if self.base_url:
            config["model_server"] = self.base_url
        if self.api_key:
            config["api_key"] = self.api_key
        config["generate_cfg"] = {
            "max_input_tokens": 32000,
            "max_tokens": 1200,
            "temperature": 0.0,
        }
        return config

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> tuple[str, LLMUsage]:
        request = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            response = self.client.chat.completions.create(
                **request,
                extra_body={
                    "enable_thinking": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        except Exception:
            response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        if usage is None:
            return content, LLMUsage()
        return (
            content,
            LLMUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            ),
        )

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> tuple[dict[str, Any], str, LLMUsage]:
        content, usage = self.chat(messages, max_tokens=max_tokens, temperature=temperature)
        parsed = _parse_json_payload(content)
        return parsed, content, usage


def _parse_json_payload(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    candidate = _extract_first_json_object(text)
    if not candidate:
        return {}
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def _sum_optional_ints(left: Optional[int], right: Optional[int]) -> Optional[int]:
    if left is None and right is None:
        return None
    return int(left or 0) + int(right or 0)
