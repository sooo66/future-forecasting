"""Forecasting runtime helpers with per-question state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

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

    def __init__(self, *, work_dir: str | Path):
        super().__init__({"work_dir": str(Path(work_dir).resolve())})
        self._initialized = False
        self._active_question_key: str | None = None

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
        return super().call(params, files=files, timeout=timeout, **kwargs)


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
