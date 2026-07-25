"""Load → validate → dedupe → convert pipeline for Part 1 (library-only core)."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from eval.part1.constants import (
    BUCKET_CONVERTED,
    BUCKET_FELL_BACK,
    BUCKET_INVALID_INPUT,
    INCLUDED_CATEGORIES,
)


def canonical_tool_json(tool: dict[str, Any]) -> str:
    """Compact, key-sorted JSON for hashing and token baselines (post-map)."""
    return json.dumps(tool, sort_keys=True, separators=(",", ":"))


def pretty_tool_json(tool: dict[str, Any]) -> str:
    """Pretty-printed normalized tool JSON (robustness baseline for token stats)."""
    return json.dumps(tool, sort_keys=True, indent=2)


def tool_hash(normalized_tool: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_tool_json(normalized_tool).encode()).hexdigest()


@dataclass
class RawToolOccurrence:
    tool: dict[str, Any]
    category: str


@dataclass
class UniqueToolAggregate:
    normalized_tool: dict[str, Any]
    tool_hash: str
    source_categories: set[str] = field(default_factory=set)
    raw_occurrence_count: int = 0
    rename_touched: bool = False


@dataclass
class UniqueToolResult:
    tool_hash: str
    source_categories: list[str]
    raw_occurrence_count: int
    bucket: str
    fallback_reason: str | None = None
    invalid_reason: str | None = None
    normalized_tool: dict[str, Any] | None = None
    cats_text: str | None = None
    assumed_closed_count: int = 0
    python_type_renames: dict[str, int] = field(default_factory=dict)
    tokens_json_tiktoken: int | None = None
    tokens_cats_tiktoken: int | None = None
    tokens_json_qwen: int | None = None
    tokens_cats_qwen: int | None = None
    tokens_json_anthropic: int | None = None
    tokens_cats_anthropic: int | None = None
    delta_tiktoken: int | None = None
    delta_qwen: int | None = None
    delta_anthropic: int | None = None


@dataclass
class ValidationStats:
    pre_dedupe_tool_count: int = 0
    tools_touched_by_rename: int = 0
    float_renames: int = 0
    dict_renames: int = 0
    tuple_renames: int = 0
    any_renames: int = 0
    invalid_after_rename: int = 0


def bfcl_data_path(category: str) -> str:
    import bfcl_eval

    return os.path.join(
        os.path.dirname(bfcl_eval.__file__),
        "data",
        f"BFCL_v4_{category}.json",
    )


def load_bfcl_tools(categories: Iterable[str] = INCLUDED_CATEGORIES) -> list[RawToolOccurrence]:
    """Read JSONL category files; collect every tool from each entry's ``function`` list."""
    out: list[RawToolOccurrence] = []
    for category in categories:
        path = bfcl_data_path(category)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                for tool in entry["function"]:
                    if not isinstance(tool, dict) or "parameters" not in tool:
                        raise AssertionError(
                            f"unexpected tool shape in {category!r}: {tool!r}"
                        )
                    out.append(
                        RawToolOccurrence(
                            tool=copy.deepcopy(tool),
                            category=category,
                        )
                    )
    return out


def validate_parameters_schema(parameters: dict[str, Any]) -> str | None:
    """Return an error string if ``parameters`` is not draft 2020-12, else None."""
    try:
        Draft202012Validator.check_schema(parameters)
    except SchemaError as exc:
        return str(exc)
    return None


def validate_and_aggregate(
    occurrences: list[RawToolOccurrence],
    *,
    normalize_map_python_types,
) -> tuple[list[UniqueToolAggregate], list[tuple[UniqueToolAggregate, str]], ValidationStats]:
    """Map Python types, validate, and build global dedupe aggregates."""
    stats = ValidationStats(pre_dedupe_tool_count=len(occurrences))
    by_hash: dict[str, UniqueToolAggregate] = {}
    invalid: list[tuple[UniqueToolAggregate, str]] = []

    for occ in occurrences:
        normalized, rename_report = normalize_map_python_types(occ.tool)
        touched = bool(rename_report.touched_locations)
        if touched:
            stats.tools_touched_by_rename += 1
        stats.float_renames += rename_report.float_count
        stats.dict_renames += rename_report.dict_count
        stats.tuple_renames += rename_report.tuple_count
        stats.any_renames += rename_report.any_count

        h = tool_hash(normalized)
        if h not in by_hash:
            by_hash[h] = UniqueToolAggregate(
                normalized_tool=normalized,
                tool_hash=h,
                rename_touched=touched,
            )
        agg = by_hash[h]
        agg.source_categories.add(occ.category)
        agg.raw_occurrence_count += 1

        reason = validate_parameters_schema(normalized["parameters"])
        if reason is not None:
            stats.invalid_after_rename += 1
            invalid.append((agg, reason))

    # Deduplicate invalid list to unique hashes (keep first reason)
    seen_invalid: set[str] = set()
    unique_invalid: list[tuple[UniqueToolAggregate, str]] = []
    for agg, reason in invalid:
        if agg.tool_hash in seen_invalid:
            continue
        seen_invalid.add(agg.tool_hash)
        unique_invalid.append((agg, reason))

    return list(by_hash.values()), unique_invalid, stats


def convert_unique_tools(
    aggregates: list[UniqueToolAggregate],
    *,
    convert_with_report,
    invalid_hashes: set[str],
    invalid_reasons: dict[str, str],
) -> list[UniqueToolResult]:
    results: list[UniqueToolResult] = []
    for agg in aggregates:
        if agg.tool_hash in invalid_hashes:
            results.append(
                UniqueToolResult(
                    tool_hash=agg.tool_hash,
                    source_categories=sorted(agg.source_categories),
                    raw_occurrence_count=agg.raw_occurrence_count,
                    bucket=BUCKET_INVALID_INPUT,
                    invalid_reason=invalid_reasons[agg.tool_hash],
                    normalized_tool=agg.normalized_tool,
                )
            )
            continue

        report = convert_with_report(
            agg.normalized_tool,
            assume_closed=True,
            map_python_types=True,
        )
        if report.fallback_count > 0:
            reason = report.fallbacks[0].reason if report.fallbacks else "unknown"
            results.append(
                UniqueToolResult(
                    tool_hash=agg.tool_hash,
                    source_categories=sorted(agg.source_categories),
                    raw_occurrence_count=agg.raw_occurrence_count,
                    bucket=BUCKET_FELL_BACK,
                    fallback_reason=reason,
                    normalized_tool=agg.normalized_tool,
                    assumed_closed_count=len(report.assumed_closed),
                    python_type_renames=_rename_counts(report.python_type_renames),
                )
            )
        else:
            results.append(
                UniqueToolResult(
                    tool_hash=agg.tool_hash,
                    source_categories=sorted(agg.source_categories),
                    raw_occurrence_count=agg.raw_occurrence_count,
                    bucket=BUCKET_CONVERTED,
                    normalized_tool=agg.normalized_tool,
                    cats_text=report.cats_text,
                    assumed_closed_count=len(report.assumed_closed),
                    python_type_renames=_rename_counts(report.python_type_renames),
                )
            )
    return results


def _rename_counts(report: Any) -> dict[str, int]:
    return {
        "float": report.float_count,
        "dict": report.dict_count,
        "tuple": report.tuple_count,
        "any": report.any_count,
    }


def attach_token_counts(
    results: list[UniqueToolResult],
    *,
    tiktoken_count,
    qwen_count,
    anthropic_counts: dict[str, dict[str, int]] | None = None,
) -> None:
    """Fill token fields on converted rows in place.

    JSON side: canonical compact bytes of the normalized tool (post-map_python_types).
    CATS side: converter CATS-text output for that tool.
    """
    for row in results:
        if row.bucket != BUCKET_CONVERTED or row.normalized_tool is None or row.cats_text is None:
            continue

        json_bytes = canonical_tool_json(row.normalized_tool)
        row.tokens_json_tiktoken = tiktoken_count(json_bytes)
        row.tokens_cats_tiktoken = tiktoken_count(row.cats_text)
        row.tokens_json_qwen = qwen_count(json_bytes)
        row.tokens_cats_qwen = qwen_count(row.cats_text)
        row.delta_tiktoken = row.tokens_cats_tiktoken - row.tokens_json_tiktoken
        row.delta_qwen = row.tokens_cats_qwen - row.tokens_json_qwen

        if anthropic_counts and row.tool_hash in anthropic_counts:
            cached = anthropic_counts[row.tool_hash]
            row.tokens_json_anthropic = cached["tokens_json_anthropic"]
            row.tokens_cats_anthropic = cached["tokens_cats_anthropic"]
            row.delta_anthropic = (
                row.tokens_cats_anthropic - row.tokens_json_anthropic
            )
        else:
            row.tokens_json_anthropic = None
            row.tokens_cats_anthropic = None
            row.delta_anthropic = None


def result_to_jsonl_dict(row: UniqueToolResult) -> dict[str, Any]:
    return {
        "tool_hash": row.tool_hash,
        "source_categories": row.source_categories,
        "raw_occurrence_count": row.raw_occurrence_count,
        "bucket": row.bucket,
        "fallback_reason": row.fallback_reason,
        "invalid_reason": row.invalid_reason,
        "tokens_json_tiktoken": row.tokens_json_tiktoken,
        "tokens_cats_tiktoken": row.tokens_cats_tiktoken,
        "tokens_json_qwen": row.tokens_json_qwen,
        "tokens_cats_qwen": row.tokens_cats_qwen,
        "tokens_json_anthropic": row.tokens_json_anthropic,
        "tokens_cats_anthropic": row.tokens_cats_anthropic,
        "delta_tiktoken": row.delta_tiktoken,
        "delta_qwen": row.delta_qwen,
        "delta_anthropic": row.delta_anthropic,
    }


def build_summary(
    results: list[UniqueToolResult],
    validation_stats: ValidationStats,
) -> dict[str, Any]:
    converted = sum(1 for r in results if r.bucket == BUCKET_CONVERTED)
    fell_back = sum(1 for r in results if r.bucket == BUCKET_FELL_BACK)
    invalid = sum(1 for r in results if r.bucket == BUCKET_INVALID_INPUT)
    denom = converted + fell_back
    coverage = converted / denom if denom else 0.0

    fallback_causes = Counter(
        r.fallback_reason for r in results if r.bucket == BUCKET_FELL_BACK and r.fallback_reason
    )

    pre_dedupe = validation_stats.pre_dedupe_tool_count
    invalid_pct_pre_dedupe = (
        validation_stats.invalid_after_rename / pre_dedupe * 100 if pre_dedupe else 0.0
    )

    total_raw_occurrences = sum(r.raw_occurrence_count for r in results)

    def token_summary(attr_delta: str, attr_json: str) -> dict[str, float]:
        pairs = [
            (getattr(r, attr_json), getattr(r, attr_delta))
            for r in results
            if r.bucket == BUCKET_CONVERTED
            and getattr(r, attr_json, None) is not None
            and getattr(r, attr_delta, None) is not None
        ]
        if not pairs:
            return {"mean_delta": 0.0, "median_delta": 0.0, "mean_pct_reduction": 0.0}
        deltas = [d for _, d in pairs]
        pct = [(j - (j + d)) / j * 100 for j, d in pairs if j > 0]
        return {
            "mean_delta": statistics.mean(deltas),
            "median_delta": statistics.median(deltas),
            "mean_pct_reduction": statistics.mean(pct) if pct else 0.0,
        }

    return {
        "unique_def_count": len(results),
        "pre_dedupe_tool_occurrences": pre_dedupe,
        "total_raw_occurrences_after_dedupe_weight": total_raw_occurrences,
        "buckets": {
            BUCKET_CONVERTED: converted,
            BUCKET_FELL_BACK: fell_back,
            BUCKET_INVALID_INPUT: invalid,
        },
        "coverage": coverage,
        "invalid_input_pct_of_pre_dedupe": invalid_pct_pre_dedupe,
        "rename_stats": {
            "tools_touched_by_rename": validation_stats.tools_touched_by_rename,
            "per_alias_counts": {
                "float": validation_stats.float_renames,
                "dict": validation_stats.dict_renames,
                "tuple": validation_stats.tuple_renames,
                "any": validation_stats.any_renames,
            },
        },
        "fallback_cause_breakdown": dict(fallback_causes),
        "token_deltas": {
            "tiktoken": token_summary("delta_tiktoken", "tokens_json_tiktoken"),
            "qwen": token_summary("delta_qwen", "tokens_json_qwen"),
            "anthropic": token_summary("delta_anthropic", "tokens_json_anthropic"),
        },
    }
