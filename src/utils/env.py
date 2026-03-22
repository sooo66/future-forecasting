"""Environment loading helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable


_ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_dotenv(path: str | Path, *, override: bool = False) -> dict[str, str]:
    """Load a .env file into os.environ."""
    env_path = Path(path)
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not _ENV_KEY_PATTERN.match(key):
            continue
        value = _strip_quotes(value)
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def get_first_env(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.getenv(key, "").strip()
        if value:
            return value
    return default


def as_bool_env(*keys: str, default: bool = False) -> bool:
    value = get_first_env(*keys)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def exportable_env_lines(keys: Iterable[str]) -> list[str]:
    lines = []
    for key in keys:
        value = os.getenv(key)
        if value:
            lines.append(f"{key}={value}")
    return lines


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value

