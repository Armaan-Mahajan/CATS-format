#!/usr/bin/env python3
"""
converter_demo.py — show the full CATS round trip and the oracle's meaning verdict.

Usage (from repo root):
    uv run --project cats-converter python converter_demo.py '<json schema>'
    echo '<json schema>' | uv run --project cats-converter python converter_demo.py

Uses the Part 1 / tool-calling conversion flags (``assume_closed``,
``map_python_types``) — same as the eval pipeline and preview UI — then asks
the behavioral oracle (tests/oracle.py) whether the recovered schema accepts and
rejects the same JSON values as the input.

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


def pipeline_round_trip(schema: object) -> object:
    """JSON -> CATS text -> JSON through the public tool-calling conversion path."""
    result = cats.convert_with_report_for_tool_calling(schema)
    reparsed = parse_text(result.cats_text)
    validate(reparsed)
    return to_json(reparsed)


def main() -> None:
    raw = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    schema = load_schema(raw)

    result = cats.convert_with_report_for_tool_calling(schema)

    print("--- JSON → CATS ---")
    print()
    print("```cats")
    print(result.cats_text)
    print("```")
    print()
    print("--- Oracle: Meaning Preservation ---")

    oracle.round_trip = lambda _ignored, _original=schema: pipeline_round_trip(_original)
    try:
        report = oracle.assert_round_trip_preserves_meaning(schema)
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
