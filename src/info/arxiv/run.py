"""Run ArXiv importer via src/info/arxiv path."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from info.paper.run import main


if __name__ == "__main__":
    main()
