"""Forecasting runtime helpers with per-question state."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import json5
from qwen_agent.tools.base import BaseTool
from qwen_agent.tools.code_interpreter import CodeInterpreter


_BASELINE_GLOBALS_NAME = "_FF_BASELINE_GLOBALS"
_RESIDENT_INIT_CODE = (
    "import matplotlib.pyplot as plt\n"
    f"{_BASELINE_GLOBALS_NAME} = set(globals().keys())\n"
    "plt.close('all')\n"
)
_RESIDENT_RESET_CODE = (
    "import gc\n"
    "import matplotlib.pyplot as plt\n"
    f"_ff_keep = set({_BASELINE_GLOBALS_NAME}) | {{{_BASELINE_GLOBALS_NAME!r}, '_ff_keep'}}\n"
    "for _ff_name in list(globals().keys()):\n"
    "    if _ff_name in _ff_keep:\n"
    "        continue\n"
    "    try:\n"
    "        del globals()[_ff_name]\n"
    "    except Exception:\n"
    "        pass\n"
    "plt.close('all')\n"
    "gc.collect()\n"
)


class ResidentCodeInterpreterTool(CodeInterpreter):
    """Reuse one code-interpreter kernel within a method run while resetting state per question."""

    description = (
        "Python 代码沙盒。参数必须是 JSON 对象，且只包含一个字符串字段："
        ' {"code": "..."}。不要使用 Markdown 代码块，不要附加解释文字。'
    )
    parameters = [{"name": "code", "type": "string", "description": "待执行的 Python 代码", "required": True}]

    def __init__(self, *, work_dir: str | Path):
        super().__init__({"work_dir": str(Path(work_dir).resolve())})
        self._initialized = False
        self._active_question_key: str | None = None

    @property
    def args_format(self) -> str:
        return '将参数格式化为 JSON 对象，且只包含一个字段：{"code": "..."}。不要使用 Markdown 代码块。'

    def begin_question(self, question_key: str) -> None:
        if self._active_question_key == question_key:
            return
        if self._initialized:
            super().call(json.dumps({"code": _RESIDENT_RESET_CODE}), timeout=20)
        self._active_question_key = question_key

    def call(self, params: str | dict, files: list[str] | None = None, timeout: Optional[int] = 30, **kwargs: Any) -> str:
        if not self._initialized:
            super().call(json.dumps({"code": _RESIDENT_INIT_CODE}), timeout=20)
            self._initialized = True
        normalized = _normalize_code_interpreter_params(params)
        if normalized is None:
            return (
                "Malformed code_interpreter arguments. "
                'Call code_interpreter again with a valid JSON object of the form {"code": "..."} only.'
            )
        return super().call(json.dumps(normalized, ensure_ascii=False), files=files, timeout=timeout, **kwargs)


class FlexMemoryTool(BaseTool):
    name = "memory"
    description = (
        "Retrieve prior experiences from the evolving FLEX library. "
        "You can filter by zone (golden or warning) and level (strategy, pattern, case)."
    )
    parameters = [
        {"name": "query", "type": "string", "description": "Current reasoning query or state.", "required": True},
        {"name": "zone", "type": "string", "description": "Optional zone filter: golden or warning.", "required": False},
        {
            "name": "level",
            "type": "string",
            "description": "Optional level filter: strategy, pattern, or case.",
            "required": False,
        },
        {"name": "top_k", "type": "integer", "description": "Maximum number of memories to return.", "required": False},
    ]

    def __init__(self, library: Any, *, cutoff_time: str):
        super().__init__()
        self._library = library
        self.cutoff_time = cutoff_time

    def call(self, params: str | dict, **kwargs: Any) -> dict[str, Any]:
        payload = self._verify_json_format_args(params)
        top_k = max(1, min(int(payload.get("top_k") or 5), 10))
        hits = self._library.retrieve(
            str(payload.get("query") or ""),
            open_time=self.cutoff_time,
            top_k=top_k,
            zone=str(payload.get("zone") or "").strip().lower() or None,
            level=str(payload.get("level") or "").strip().lower() or None,
        )
        return {
            "cutoff_time": self.cutoff_time,
            "count": len(hits),
            "hits": [item.to_tool_dict() for item in hits],
        }

    def format_result_for_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        hits = payload.get("hits")
        if not isinstance(hits, list):
            return payload
        return {
            "count": len(hits),
            "hits": [
                {
                    "zone": hit.get("zone"),
                    "level": hit.get("level"),
                    "title": hit.get("title"),
                    "summary": hit.get("summary"),
                    "content": hit.get("content"),
                }
                for hit in hits
            ],
        }


def _normalize_code_interpreter_params(params: str | dict) -> dict[str, str] | None:
    if isinstance(params, dict):
        code = params.get("code")
        if isinstance(code, str) and code.strip():
            return {"code": code}
        return None
    text = str(params or "").strip()
    if not text:
        return None

    direct = _extract_code_from_json_payload(text)
    if direct is not None:
        return {"code": direct}

    fenced = _extract_code_block(text)
    if fenced is not None:
        return {"code": fenced}

    recovered = _recover_truncated_code_payload(text)
    if recovered is not None:
        return {"code": recovered}
    return None


def _extract_code_from_json_payload(text: str) -> str | None:
    try:
        payload = json5.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    code = payload.get("code")
    if not isinstance(code, str):
        return None
    code = code.strip()
    return code or None


def _extract_code_block(text: str) -> str | None:
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    code = match.group(1).strip()
    return code or None


def _recover_truncated_code_payload(text: str) -> str | None:
    match = re.search(r'["\']code["\']\s*:\s*', text)
    if not match:
        return None
    remainder = text[match.end():].lstrip()
    if not remainder or remainder[0] not in {'"', "'"}:
        return None
    quote = remainder[0]
    body = remainder[1:]

    terminated_index = _find_unescaped_quote(body, quote)
    candidates: list[str] = []
    if terminated_index is not None:
        candidates.append(body[: terminated_index + 1])

    trimmed = body.rstrip()
    candidates.append(trimmed + quote)
    stripped_backslashes = trimmed.rstrip("\\")
    if stripped_backslashes != trimmed:
        candidates.append(stripped_backslashes + quote)

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            code = json5.loads(quote + candidate)
        except Exception:
            continue
        if isinstance(code, str):
            code = code.strip()
            if code:
                return code
    return None


def _find_unescaped_quote(text: str, quote: str) -> int | None:
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == quote:
            return index
    return None
