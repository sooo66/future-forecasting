"""Custom tools live here."""

from __future__ import annotations

from typing import Any

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool("echo")
class EchoTool(BaseTool):
    description = "Echo back the input text."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def call(self, params: str | dict, **kwargs: Any) -> str:
        payload = self._verify_json_format_args(params)
        return str(payload.get("text", ""))

