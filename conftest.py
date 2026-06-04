"""Pytest configuration. Adds the repo root to sys.path so the
``shared`` and ``services`` packages are importable when tests are run
from any working directory (``pytest`` does not add the root by default)."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
