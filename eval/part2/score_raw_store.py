"""Score Part 2 eval cells from the immutable raw store."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.part2.checker import (
    PART2_CLAUDE_MODEL,
    PART2_OPENAI_MODEL,
    PART2_QWEN_MODEL,
    Part2Condition,
    bfcl_model_name_for_checker,
)
from eval.part2.corpus import CorpusEntry, load_part2_corpus
from eval.part2.pilot_entries import PILOT_ENTRY_ALL_FALLBACK
from eval.part2.providers import Provider
from eval.part2.response_parser import ExtractedModelOutput, extract_model_output
from eval.part2.scoring import score_extracted_response


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _provider_enum(provider: str) -> Provider:
    if provider == "openai":
        return Provider.OPENAI
    if provider == "anthropic":
        return Provider.ANTHROPIC
    if provider == "qwen":
        return Provider.OPENROUTER
    raise ValueError(f"unknown provider: {provider!r}")


@dataclass(frozen=True)
class ScoredCell:
    custom_id: str
    entry_id: str
    category: str
    model: str
    condition: str
    provider: str
    status: str
    syntactic_valid: bool
    semantic_correct: bool | None
    parse_outcome: str | None
    raw_response_ref: str | None
    scored_via: str | None
    skip_reason: str | None
    api_error: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "custom_id": self.custom_id,
            "entry_id": self.entry_id,
            "category": self.category,
            "model": self.model,
            "condition": self.condition,
            "provider": self.provider,
            "status": self.status,
            "syntactic_valid": self.syntactic_valid,
            "semantic_correct": self.semantic_correct,
            "parse_outcome": self.parse_outcome,
            "raw_response_ref": self.raw_response_ref,
            "scored_via": self.scored_via,
            "skip_reason": self.skip_reason,
            "api_error": self.api_error,
        }


def score_raw_record(record: dict[str, Any], entry: CorpusEntry) -> ScoredCell:
    custom_id = record["custom_id"]
    model = str(record["model"])
    condition = Part2Condition(str(record["condition"]))
    raw_ref = record.get("raw_response_ref")
    status = record["status"]

    if status == "skipped":
        return ScoredCell(
            custom_id=custom_id,
            entry_id=record["entry_id"],
            category=record["category"],
            model=model,
            condition=condition.value,
            provider=record["provider"],
            status="skipped",
            syntactic_valid=False,
            semantic_correct=None,
            parse_outcome=None,
            raw_response_ref=raw_ref,
            scored_via=None,
            skip_reason=record.get("skip_reason"),
            api_error=None,
        )

    if status == "api_error":
        error = record.get("error")
        if isinstance(error, dict):
            error = json.dumps(error, ensure_ascii=False)
        return ScoredCell(
            custom_id=custom_id,
            entry_id=record["entry_id"],
            category=record["category"],
            model=model,
            condition=condition.value,
            provider=record["provider"],
            status="api_error",
            syntactic_valid=False,
            semantic_correct=None,
            parse_outcome=None,
            raw_response_ref=raw_ref,
            scored_via=None,
            skip_reason=None,
            api_error=str(error) if error is not None else "api_error",
        )

    scored_via = bfcl_model_name_for_checker(model, condition)
    raw = record.get("raw_response")
    if not isinstance(raw, dict):
        return ScoredCell(
            custom_id=custom_id,
            entry_id=record["entry_id"],
            category=record["category"],
            model=model,
            condition=condition.value,
            provider=record["provider"],
            status="completed",
            syntactic_valid=False,
            semantic_correct=None,
            parse_outcome="missing_raw_response",
            raw_response_ref=raw_ref,
            scored_via=scored_via,
            skip_reason=None,
            api_error=None,
        )

    try:
        extracted = extract_model_output(
            provider=_provider_enum(str(record["provider"])),
            condition=condition,
            response=raw,
        )
    except Exception as exc:  # noqa: BLE001
        extracted = ExtractedModelOutput(
            prompt_text=None,
            model_output=None,
            parse_outcome=None,
            response_tool_name=None,
        )
        return ScoredCell(
            custom_id=custom_id,
            entry_id=record["entry_id"],
            category=record["category"],
            model=model,
            condition=condition.value,
            provider=record["provider"],
            status="completed",
            syntactic_valid=False,
            semantic_correct=None,
            parse_outcome=f"extraction_error:{type(exc).__name__}",
            raw_response_ref=raw_ref,
            scored_via=scored_via,
            skip_reason=None,
            api_error=None,
        )

    scored = score_extracted_response(
        entry,
        model=model,
        condition=condition,
        extracted=extracted,
        skipped=False,
    )
    return ScoredCell(
        custom_id=custom_id,
        entry_id=record["entry_id"],
        category=record["category"],
        model=model,
        condition=condition.value,
        provider=record["provider"],
        status="completed",
        syntactic_valid=scored.syntactically_valid,
        semantic_correct=scored.semantically_valid if scored.syntactically_valid else None,
        parse_outcome=scored.parse_outcome.value if scored.parse_outcome else None,
        raw_response_ref=raw_ref,
        scored_via=scored_via,
        skip_reason=None,
        api_error=None,
    )


def summarize_scored_cells(cells: list[ScoredCell]) -> dict[str, dict[str, Any]]:
    """Per model × condition summary over completed cells only."""
    buckets: dict[tuple[str, str], list[ScoredCell]] = defaultdict(list)
    for cell in cells:
        if cell.status != "completed":
            continue
        buckets[(cell.model, cell.condition)].append(cell)

    summary: dict[str, dict[str, Any]] = {}
    for (model, condition), rows in sorted(buckets.items()):
        n = len(rows)
        syn = sum(1 for row in rows if row.syntactic_valid)
        sem = sum(1 for row in rows if row.semantic_correct is True)
        syn_den = syn
        summary[f"{model}|{condition}"] = {
            "model": model,
            "condition": condition,
            "n": n,
            "syntactic_valid_count": syn,
            "syntactic_valid_rate": round(syn / n, 4) if n else None,
            "semantic_correct_count": sem,
            "semantic_correct_rate": round(sem / syn_den, 4) if syn_den else None,
        }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    fieldnames = [
        "custom_id",
        "entry_id",
        "category",
        "model",
        "condition",
        "provider",
        "status",
        "syntactic_valid",
        "semantic_correct",
        "parse_outcome",
        "raw_response_ref",
        "scored_via",
        "skip_reason",
        "api_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


STATS_TASK_NOTE = (
    "McNemar paired (a)-vs-(b): drop entry "
    f"{PILOT_ENTRY_ALL_FALLBACK!r} — it has no condition (a) cell (cats_all_fallback skip). "
    "Do not treat as tie or miss."
)
