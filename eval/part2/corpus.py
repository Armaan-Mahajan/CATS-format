"""Load Part 2 BFCL corpus entries paired with ground truth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.part2.paths import bfcl_ground_truth_path, bfcl_prompt_path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSAMPLE_PATH = REPO_ROOT / "eval" / "part2" / "live_multiple_subsample.json"

PART2_CATEGORIES = ("live_simple", "multiple", "live_multiple")

EXPECTED_ENTRY_COUNTS: dict[str, int] = {
    "live_simple": 258,
    "multiple": 200,
    "live_multiple": 527,
}


@dataclass(frozen=True)
class CorpusEntry:
    """One BFCL test case: prompt, tools, and AST ground truth."""

    id: str
    category: str
    question: list[Any]
    function: list[dict[str, Any]]
    ground_truth: list[dict[str, Any]]


class CorpusLoadError(Exception):
    """Raised when prompt/ground-truth pairing or counts are inconsistent."""


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise CorpusLoadError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return entries


def _load_ground_truth_by_id(category: str) -> dict[str, list[dict[str, Any]]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(bfcl_ground_truth_path(category)):
        entry_id = row["id"]
        if entry_id in by_id:
            raise CorpusLoadError(
                f"duplicate ground-truth id {entry_id!r} in {category!r}"
            )
        if "ground_truth" not in row:
            raise CorpusLoadError(
                f"ground-truth row for {entry_id!r} missing 'ground_truth' field"
            )
        by_id[entry_id] = row["ground_truth"]
    return by_id


def load_live_multiple_subsample_ids() -> set[str]:
    with open(SUBSAMPLE_PATH, encoding="utf-8") as handle:
        payload = json.load(handle)
    selected = payload["selected_ids"]
    if len(selected) != payload["n"]:
        raise CorpusLoadError(
            f"subsample n={payload['n']} but len(selected_ids)={len(selected)}"
        )
    return set(selected)


def load_category_entries(
    category: str,
    *,
    id_filter: set[str] | None = None,
) -> list[CorpusEntry]:
    """Load prompt entries for ``category``, joined to ground truth on ``id``."""
    if category not in PART2_CATEGORIES:
        raise ValueError(f"unsupported Part 2 category: {category!r}")
    if category == "live_multiple" and id_filter is None:
        raise ValueError(
            "live_multiple requires id_filter=load_live_multiple_subsample_ids()"
        )

    ground_truth_by_id = _load_ground_truth_by_id(category)
    entries: list[CorpusEntry] = []

    for row in _read_jsonl(bfcl_prompt_path(category)):
        entry_id = row["id"]
        if id_filter is not None and entry_id not in id_filter:
            continue
        if entry_id not in ground_truth_by_id:
            raise CorpusLoadError(
                f"prompt entry {entry_id!r} in {category!r} has no ground truth"
            )
        for field in ("question", "function"):
            if field not in row:
                raise CorpusLoadError(
                    f"prompt entry {entry_id!r} in {category!r} missing {field!r}"
                )
        entries.append(
            CorpusEntry(
                id=entry_id,
                category=category,
                question=row["question"],
                function=row["function"],
                ground_truth=ground_truth_by_id[entry_id],
            )
        )

    if id_filter is not None:
        loaded_ids = {entry.id for entry in entries}
        missing = sorted(id_filter - loaded_ids)
        if missing:
            raise CorpusLoadError(
                f"{category!r}: {len(missing)} subsample id(s) missing from prompt file "
                f"(first: {missing[0]!r})"
            )

    expected = EXPECTED_ENTRY_COUNTS[category]
    if len(entries) != expected:
        raise CorpusLoadError(
            f"{category!r}: expected {expected} entries after pairing, got {len(entries)}"
        )

    return entries


def load_part2_corpus() -> list[CorpusEntry]:
    """Load all 985 Part 2 entries (258 + 200 + 527) with ground truth."""
    subsample_ids = load_live_multiple_subsample_ids()
    corpus: list[CorpusEntry] = []
    corpus.extend(load_category_entries("live_simple"))
    corpus.extend(load_category_entries("multiple"))
    corpus.extend(load_category_entries("live_multiple", id_filter=subsample_ids))
    return corpus


def category_counts() -> dict[str, int]:
    """Return entry counts per category without building full CorpusEntry objects."""
    subsample_ids = load_live_multiple_subsample_ids()
    return {
        "live_simple": len(load_category_entries("live_simple")),
        "multiple": len(load_category_entries("multiple")),
        "live_multiple": len(
            load_category_entries("live_multiple", id_filter=subsample_ids)
        ),
    }
