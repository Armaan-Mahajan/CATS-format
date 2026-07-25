#!/usr/bin/env python3
"""
smoke_test.py — human-runnable end-to-end sanity check via the public ``cats`` API.

Run from cats-converter/:

    uv run python smoke_test.py

Uses tests/oracle.py for behavioral meaning checks (same harness as the test suite).
"""

from __future__ import annotations

import copy
import os
import sys
from typing import Any

# Oracle lives under tests/ (test infrastructure, not part of the public API).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))

import oracle  # noqa: E402

import cats  # noqa: E402

JsonSchema = dict[str, Any] | list[Any]

# --- Representative schemas ------------------------------------------------

MULTI_FEATURE_TOOL: dict[str, Any] = {
    "name": "create_event",
    "description": "Schedule a calendar event",
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 1, "maxLength": 200},
        "priority": {"type": "integer", "enum": [1, 2, 3]},
        "all_day": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "location": {"$ref": "#/$defs/Venue"},
    },
    "required": ["title", "priority"],
    "additionalProperties": False,
    "$defs": {
        "Venue": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}

OPENAI_WRAPPED = {
    "name": "assess_answer",
    "description": "Assess a student's answer and assign a score",
    "parameters": {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 0, "maximum": 100},
            "feedback": {"type": "string", "minLength": 1},
        },
        "required": ["score"],
        "additionalProperties": False,
    },
    "strict": True,
}

ARRAY_OBJECT_NESTED = {
    "name": "batch_update",
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer", "minimum": 1}},
                "required": ["id"],
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}

FALLBACK_NOT = {
    "name": "reject_strings",
    "type": "object",
    "properties": {"q": {"not": {"type": "string"}}},
    "additionalProperties": False,
}

LOW_CONFIDENCE_CONST = {
    "name": "mode",
    "type": "object",
    "properties": {"mode": {"const": "automatic"}},
    "required": ["mode"],
    "additionalProperties": False,
}

LOW_CONFIDENCE_PATTERN = {
    "name": "token",
    "type": "object",
    "properties": {
        "token": {
            "type": "string",
            "pattern": "^[A-Z]{3}-[0-9]{4}$",
            "minLength": 8,
            "maxLength": 8,
        }
    },
    "required": ["token"],
    "additionalProperties": False,
}


def _tool_calling_round_trip(schema: JsonSchema) -> Any:
    """JSON Schema -> CATS text -> JSON Schema (Part 1 / tool-calling flags)."""
    text = cats.convert_for_tool_calling(copy.deepcopy(schema))
    return cats.to_json_schema(text)


def _run_case(name: str, schema: JsonSchema, *, extra_instances: list[Any] | None = None) -> None:
    print("=" * 72)
    print(f"Schema: {name}")
    print("-" * 72)

    result = cats.convert_with_report_for_tool_calling(copy.deepcopy(schema))
    encoded = result.fallback_count == 0
    print(f"Encoded: {'yes' if encoded else 'no (fallback)'}")
    if result.fallback_count:
        for fb in result.fallbacks:
            loc = fb.location or "(root)"
            print(f"  fallback @ {loc}: {fb.reason}")
    if result.warnings:
        print(f"Validation warnings: {len(result.warnings)}")
        for w in result.warnings[:3]:
            print(f"  - {w}")
        if len(result.warnings) > 3:
            print(f"  ... and {len(result.warnings) - 3} more")

    print()
    print("CATS output:")
    print(result.cats_text)
    print()

    saved_round_trip = oracle.round_trip
    oracle.round_trip = _tool_calling_round_trip
    try:
        report = oracle.assert_round_trip_preserves_meaning(
            schema,
            extra_instances=extra_instances or [],
        )
        verdict = "PASS (meaning preserved)"
        detail = report.note
        if report.low_confidence:
            detail = f"{detail} [LOW CONFIDENCE]"
    except AssertionError as exc:
        verdict = "FAIL (meaning changed)"
        detail = str(exc).split("\n", 1)[0]
    finally:
        oracle.round_trip = saved_round_trip

    path = "tool-calling API (assume_closed + map_python_types)"
    print(f"Oracle ({path}): {verdict}")
    print(f"  {detail}")
    print()


def main() -> None:
    print("CATS converter smoke test (public API + behavioral oracle)\n")

    _run_case(
        "multi_feature_tool",
        MULTI_FEATURE_TOOL,
        extra_instances=[
            {"title": "Meet", "priority": 2},
            {"title": "Meet", "priority": 2, "location": {"name": "Hall"}},
            {"title": "", "priority": 1},
            {"priority": 1},
            {"title": "x", "priority": 99},
        ],
    )
    _run_case(
        "openai_parameters_wrapper",
        OPENAI_WRAPPED,
        extra_instances=[
            {"score": 50},
            {"score": -1},
            {"score": 50, "feedback": "ok"},
            {},
        ],
    )
    _run_case(
        "array_object_nested",
        ARRAY_OBJECT_NESTED,
        extra_instances=[
            {"items": [{"id": 1}]},
            {"items": []},
            {"items": [{"id": 0}]},
            {},
        ],
    )
    _run_case(
        "fallback_not",
        FALLBACK_NOT,
        extra_instances=[{"q": 1}, {"q": "s"}, {}],
    )
    _run_case(
        "low_confidence_const",
        LOW_CONFIDENCE_CONST,
        extra_instances=[{"mode": "automatic"}, {"mode": "manual"}, {}],
    )
    _run_case(
        "low_confidence_tight_pattern",
        LOW_CONFIDENCE_PATTERN,
        extra_instances=[
            {"token": "ABC-1234"},
            {"token": "abc-1234"},
            {"token": "TOO-LONG"},
            {},
        ],
    )

    print("=" * 72)
    print("Smoke run complete.")


if __name__ == "__main__":
    main()
