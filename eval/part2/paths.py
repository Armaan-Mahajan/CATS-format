"""BFCL data path resolution for Part 2 (entry-level, not per-tool)."""

from __future__ import annotations

import os


def bfcl_package_root() -> str:
    import bfcl_eval

    return os.path.dirname(bfcl_eval.__file__)


def bfcl_prompt_path(category: str) -> str:
    return os.path.join(bfcl_package_root(), "data", f"BFCL_v4_{category}.json")


def bfcl_ground_truth_path(category: str) -> str:
    return os.path.join(
        bfcl_package_root(), "data", "possible_answer", f"BFCL_v4_{category}.json"
    )
