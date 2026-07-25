#!/usr/bin/env python3
"""Part 1 eval CLI: BFCL conversion coverage + per-tool token cost.

Run from repo root::

    uv run --project cats-converter python eval/run_part1.py --output-dir eval/out

Anthropic column (one-time, cached)::

    uv run --project cats-converter python eval/run_part1.py \\
        --output-dir eval/out --with-anthropic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
CATS_CONVERTER = REPO_ROOT / "cats-converter"
sys.path.insert(0, str(CATS_CONVERTER))
sys.path.insert(0, str(REPO_ROOT))

from eval.part1.anthropic_cache import AnthropicTokenCache  # noqa: E402
from eval.part1.anthropic_fetch import fetch_anthropic_compact_counts  # noqa: E402
from eval.part1.constants import BUCKET_CONVERTED  # noqa: E402
from eval.part1.pipeline import (  # noqa: E402
    attach_token_counts,
    build_summary,
    canonical_tool_json,
    convert_unique_tools,
    load_bfcl_tools,
    result_to_jsonl_dict,
    validate_and_aggregate,
)
from eval.part1.tokenizers import (  # noqa: E402
    default_qwen_counter,
    default_tiktoken_counter,
    ensure_hf_cache_in_repo,
)

import cats  # noqa: E402
from from_json import normalize_map_python_types  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="CATS Part 1 eval (BFCL coverage + tokens)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "eval" / "out",
        help="Directory for records.jsonl and summary.json",
    )
    parser.add_argument(
        "--with-anthropic",
        action="store_true",
        help="Fetch Anthropic token counts (cached; requires API key)",
    )
    parser.add_argument(
        "--anthropic-workers",
        type=int,
        default=2,
        help="Concurrent Anthropic count_tokens workers (global rate limit is the bottleneck)",
    )
    args = parser.parse_args()

    ensure_hf_cache_in_repo(REPO_ROOT)

    print("Loading BFCL tools…")
    occurrences = load_bfcl_tools()
    print(f"  pre-dedupe tool occurrences: {len(occurrences)}")

    aggregates, invalid_entries, validation_stats = validate_and_aggregate(
        occurrences,
        normalize_map_python_types=normalize_map_python_types,
    )
    invalid_hashes = {agg.tool_hash for agg, _ in invalid_entries}
    invalid_reasons = {agg.tool_hash: reason for agg, reason in invalid_entries}

    print("Converting unique defs (assume_closed=True, map_python_types=True)…")
    results = convert_unique_tools(
        aggregates,
        convert_with_report=cats.convert_with_report,
        invalid_hashes=invalid_hashes,
        invalid_reasons=invalid_reasons,
    )

    tik = default_tiktoken_counter()
    qwen = default_qwen_counter()

    anthropic_map: dict[str, dict[str, int]] = {}
    cache_path = REPO_ROOT / "eval" / "cache" / "anthropic_tokens.json"
    cache = AnthropicTokenCache(cache_path)

    if args.with_anthropic:
        print("Fetching Anthropic token counts (cached)…")
        eligible = [
            row
            for row in results
            if row.bucket == BUCKET_CONVERTED
            and row.normalized_tool is not None
            and row.cats_text is not None
        ]
        fetch_anthropic_compact_counts(
            eligible,
            cache,
            json_text_for=canonical_tool_json,
            max_workers=args.anthropic_workers,
        )
        cache.save()

    for row in results:
        if row.bucket != BUCKET_CONVERTED:
            continue
        cached = cache.get(row.tool_hash)
        if cached is not None:
            anthropic_map[row.tool_hash] = cached

    attach_token_counts(
        results,
        tiktoken_count=tik.count,
        qwen_count=qwen.count,
        anthropic_counts=anthropic_map or None,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    summary_path = args.output_dir / "summary.json"

    with records_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(result_to_jsonl_dict(row), separators=(",", ":")))
            handle.write("\n")

    summary = build_summary(results, validation_stats)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {records_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
