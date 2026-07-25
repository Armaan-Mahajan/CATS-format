"""Stable cell enumeration for the full Part 2 eval run."""

from __future__ import annotations

from dataclasses import dataclass

from eval.part2.checker import (
    PART2_CLAUDE_MODEL,
    PART2_OPENAI_MODEL,
    PART2_QWEN_MODEL,
    Part2Condition,
)
from eval.part2.corpus import CorpusEntry, load_part2_corpus
from eval.part2.pilot_entries import PILOT_ENTRY_ALL_FALLBACK

ALL_FALLBACK_ENTRY_ID = PILOT_ENTRY_ALL_FALLBACK

OPENAI_BATCH_REQUESTS = 2954
ANTHROPIC_BATCH_REQUESTS = 2954
QWEN_SYNC_REQUESTS = 2954
TOTAL_LOGICAL_CELLS = 985 * 3 * 3  # 8865, includes 3 skipped (a) slots
API_CELLS_PER_PROVIDER = 2954

# Anthropic: ^[a-zA-Z0-9_-]{1,64}$ per official API reference.
# OpenAI: custom_id max 512 chars (platform.openai.com/docs/api-reference/batch).
CUSTOM_ID_PREFIX = "c"
CUSTOM_ID_WIDTH = 7  # c0000001 … c0002954 (9 chars)


@dataclass(frozen=True)
class CellSpec:
    custom_id: str
    entry_id: str
    category: str
    condition: Part2Condition
    model: str
    skipped: bool
    skip_reason: str | None = None


def _should_skip_cats(entry_id: str, condition: Part2Condition) -> bool:
    return entry_id == ALL_FALLBACK_ENTRY_ID and condition == Part2Condition.CATS_IN_PROMPT


def build_cell_plan(corpus: list[CorpusEntry] | None = None) -> list[CellSpec]:
    """Deterministic plan: sorted entries × conditions × models."""
    if corpus is None:
        corpus = load_part2_corpus()
    entries = sorted(corpus, key=lambda entry: entry.id)
    conditions = list(Part2Condition)
    models = [PART2_OPENAI_MODEL, PART2_CLAUDE_MODEL, PART2_QWEN_MODEL]

    plan: list[CellSpec] = []
    index = 0
    for entry in entries:
        for condition in conditions:
            for model in models:
                index += 1
                skipped = _should_skip_cats(entry.id, condition)
                plan.append(
                    CellSpec(
                        custom_id=f"{CUSTOM_ID_PREFIX}{index:0{CUSTOM_ID_WIDTH}d}",
                        entry_id=entry.id,
                        category=entry.category,
                        condition=condition,
                        model=model,
                        skipped=skipped,
                        skip_reason="cats_all_fallback" if skipped else None,
                    )
                )
    return plan


def api_cells_for_provider(plan: list[CellSpec], provider_model: str) -> list[CellSpec]:
    return [
        cell
        for cell in plan
        if cell.model == provider_model and not cell.skipped
    ]


def sidecar_records(plan: list[CellSpec]) -> dict[str, dict[str, str | bool | None]]:
    return {
        cell.custom_id: {
            "entry_id": cell.entry_id,
            "category": cell.category,
            "condition": cell.condition.value,
            "model": cell.model,
            "skipped": cell.skipped,
            "skip_reason": cell.skip_reason,
        }
        for cell in plan
    }
