"""Pytest path setup for Part 2 tests under repo-root ``eval/``."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CATS_CONVERTER = REPO_ROOT / "cats-converter"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CATS_CONVERTER) not in sys.path:
    sys.path.insert(0, str(CATS_CONVERTER))
