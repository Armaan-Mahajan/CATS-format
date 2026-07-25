#!/usr/bin/env python3
"""
converter_demo_inline.py — a fixed-schema demo of the CATS round trip and oracle.

Unlike converter_demo.py (which reads schemas from the command line or stdin),
this demo has a schema baked in for quick, repeatable testing.

Usage (from repo root):
    uv run --project cats-converter python converter_demo_inline.py

Edit SCHEMA_JSON below (paste real JSON — use true/false/null, not Python True/False).

Oracle note: this demo uses convert_with_report_for_tool_calling (assume_closed=True, map_python_types=True).
The oracle may fail on open input schemas or schemas with python-specific types; that reflects intentional narrowing, not bad CATS text.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CATS_CONVERTER = REPO_ROOT / "cats-converter"
for entry in (str(CATS_CONVERTER), str(CATS_CONVERTER / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import cats  # noqa: E402
import oracle  # noqa: E402
from from_json import load_schema  # noqa: E402
from parser import parse_text  # noqa: E402
from to_json import to_json  # noqa: E402
from validate import validate  # noqa: E402

# Paste JSON here :) 
# Note: use RFC 8259 spelling: true, false, null — not Python True/False/None.
SCHEMA_JSON = """
{
  "name": "get_weather",
  "description": "Get weather for a city",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string"},
      "days": {"type": "integer", "minimum": 1}
    },
    "required": ["city"],
    "additionalProperties": false
  }
}
"""
SCHEMA = load_schema(SCHEMA_JSON)


def pipeline_round_trip(schema: object) -> object:
    """JSON -> CATS text -> JSON through the public tool-calling conversion path."""
    result = cats.convert_with_report_for_tool_calling(schema)
    reparsed = parse_text(result.cats_text)
    validate(reparsed)
    return to_json(reparsed)


def main() -> None:
    result = cats.convert_with_report_for_tool_calling(SCHEMA)

    print("--- JSON → CATS ---")
    print()
    print("```cats")
    print(result.cats_text)
    print("```")
    print()
    print("--- Oracle: Meaning Preservation ---")

    oracle.round_trip = lambda _ignored, _original=SCHEMA: pipeline_round_trip(_original)
    try:
        report = oracle.assert_round_trip_preserves_meaning(SCHEMA)
        print(report.note)
        if report.low_confidence:
            print(
                "⚠ Low confidence (thin generation) — consider adding "
                "extra_instances for this schema"
            )
    except AssertionError as exc:
        print("✗ Meaning NOT preserved:")
        print(str(exc))

    for fallback in result.fallbacks:
        print(
            "ℹ Tool fell back to raw JSON Schema (contains out-of-scope "
            f"construct: {fallback.reason})"
        )


if __name__ == "__main__":
    main()
