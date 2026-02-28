"""Sociomedia importers and runners."""
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from .run import run_sociomedia_from_config

__all__ = ["run_sociomedia_from_config"]
