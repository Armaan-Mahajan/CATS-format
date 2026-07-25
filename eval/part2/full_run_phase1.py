#!/usr/bin/env python3
"""Phase 1 — build and validate full Part 2 run artifacts (NO API submission).

Run from cats-converter/:

    uv run python ../eval/part2/full_run_phase1.py

Halts after validation and cost estimate. Do not submit until explicit approval.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CATS_CONVERTER = REPO_ROOT / "cats-converter"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CATS_CONVERTER) not in sys.path:
    sys.path.insert(0, str(CATS_CONVERTER))

from eval.part2.batch_io import (  # noqa: E402
    ANTHROPIC_MAX_BYTES,
    ANTHROPIC_MAX_REQUESTS,
    OPENAI_MAX_BYTES,
    OPENAI_MAX_REQUESTS,
    build_anthropic_batch_requests,
    build_openai_batch_jsonl,
    build_qwen_worklist,
    write_json,
    write_jsonl_rows,
    write_text,
)
from eval.part2.cell_plan import (  # noqa: E402
    ALL_FALLBACK_ENTRY_ID,
    ANTHROPIC_BATCH_REQUESTS,
    API_CELLS_PER_PROVIDER,
    OPENAI_BATCH_REQUESTS,
    QWEN_SYNC_REQUESTS,
    api_cells_for_provider,
    build_cell_plan,
    sidecar_records,
)
from eval.part2.checker import (  # noqa: E402
    PART2_CLAUDE_MODEL,
    PART2_OPENAI_MODEL,
    PART2_QWEN_MODEL,
    Part2Condition,
)
from eval.part2.corpus import load_part2_corpus  # noqa: E402

RUN_TAG = "cats-part2-eval"
RUN_ID_PREFIX = "full_eval"

DOC_URLS = {
    "openai_batch_create": "https://platform.openai.com/docs/api-reference/batch/create",
    "openai_batch_request_input": "https://platform.openai.com/docs/api-reference/batch/request-input",
    "openai_responses": "https://platform.openai.com/docs/api-reference/responses",
    "anthropic_message_batches": "https://docs.anthropic.com/en/api/creating-message-batches",
    "openrouter_models": "https://openrouter.ai/docs/api-reference/list-available-models",
}

ANTHROPIC_CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _validate_jsonl_lines(path: Path, expected_count: int) -> dict:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != expected_count:
        raise AssertionError(f"{path.name}: expected {expected_count} lines, got {len(lines)}")
    custom_ids: list[str] = []
    for index, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{path.name}:{index}: invalid JSON: {exc}") from exc
        custom_ids.append(row["custom_id"])
    if len(set(custom_ids)) != len(custom_ids):
        raise AssertionError(f"{path.name}: duplicate custom_id values")
    return {"line_count": len(lines), "unique_custom_ids": len(set(custom_ids))}


def _validate_all_fallback(plan: list, openai_cells: list, anthropic_cells: list, qwen_cells: list) -> None:
    fallback_cells = [c for c in plan if c.entry_id == ALL_FALLBACK_ENTRY_ID]
    assert len(fallback_cells) == 9

    cats_cells = [c for c in fallback_cells if c.condition == Part2Condition.CATS_IN_PROMPT]
    assert len(cats_cells) == 3 and all(c.skipped for c in cats_cells)

    for condition in (Part2Condition.JSON_IN_PROMPT, Part2Condition.NATIVE_TOOLS):
        cond_cells = [c for c in fallback_cells if c.condition == condition]
        assert len(cond_cells) == 3 and all(not c.skipped for c in cond_cells)

    def _api_entry_conditions(cells: list) -> set[str]:
        return {c.condition.value for c in cells if c.entry_id == ALL_FALLBACK_ENTRY_ID}

    for label, cells in (
        ("openai", openai_cells),
        ("anthropic", anthropic_cells),
        ("qwen", qwen_cells),
    ):
        present = _api_entry_conditions(cells)
        if present != {"json_in_prompt", "native_tools"}:
            raise AssertionError(
                f"{label}: {ALL_FALLBACK_ENTRY_ID} must appear in (b)/(c) only, got {present!r}"
            )


def main() -> None:
    stamp = _stamp()
    run_id = f"{RUN_ID_PREFIX}_{stamp}"
    out_dir = REPO_ROOT / "eval" / "part2" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_part2_corpus()
    entry_by_id = {entry.id: entry for entry in corpus}
    plan = build_cell_plan(corpus)

    print(f"Phase 1 offline build — run_id={run_id}")
    print(f"Tag: {RUN_TAG}")
    print(f"Output: {out_dir}")
    print(f"Corpus entries: {len(corpus)}")
    print(f"Logical cells (incl. skipped): {len(plan)}")

    openai_cells = api_cells_for_provider(plan, PART2_OPENAI_MODEL)
    anthropic_cells = api_cells_for_provider(plan, PART2_CLAUDE_MODEL)
    qwen_cells = api_cells_for_provider(plan, PART2_QWEN_MODEL)

    assert len(openai_cells) == OPENAI_BATCH_REQUESTS
    assert len(anthropic_cells) == ANTHROPIC_BATCH_REQUESTS
    assert len(qwen_cells) == QWEN_SYNC_REQUESTS

    openai_jsonl, _openai_meta = build_openai_batch_jsonl(openai_cells, entry_by_id)
    anthropic_requests, _anthropic_meta = build_anthropic_batch_requests(anthropic_cells, entry_by_id)
    qwen_rows = build_qwen_worklist(qwen_cells, entry_by_id)

    openai_path = write_text(out_dir / "openai_batch_input.jsonl", openai_jsonl)
    anthropic_path = write_json(out_dir / "anthropic_batch_requests.json", {"requests": anthropic_requests})
    qwen_path = write_jsonl_rows(out_dir / "qwen_sync_worklist.jsonl", qwen_rows)
    sidecar_path = write_json(
        out_dir / "custom_id_sidecar.json",
        {
            "run_id": run_id,
            "metadata_tag": RUN_TAG,
            "cells": sidecar_records(plan),
        },
    )

    manifest = {
        "run_id": run_id,
        "metadata_tag": RUN_TAG,
        "phase": 1,
        "doc_urls": DOC_URLS,
        "counts": {
            "corpus_entries": len(corpus),
            "openai_batch_lines": OPENAI_BATCH_REQUESTS,
            "anthropic_batch_requests": ANTHROPIC_BATCH_REQUESTS,
            "qwen_sync_rows": QWEN_SYNC_REQUESTS,
            "skipped_cats_all_fallback": 3,
        },
        "paths": {
            "openai_batch_input": str(openai_path),
            "anthropic_batch_requests": str(anthropic_path),
            "qwen_sync_worklist": str(qwen_path),
            "custom_id_sidecar": str(sidecar_path),
        },
    }
    manifest_path = write_json(out_dir / "phase1_manifest.json", manifest)

    # --- validation ---
    print("\n=== Validation ===")
    openai_val = _validate_jsonl_lines(openai_path, OPENAI_BATCH_REQUESTS)
    print(f"OpenAI JSONL: {openai_val}")

    anthropic_ids = [row["custom_id"] for row in anthropic_requests]
    if len(anthropic_ids) != ANTHROPIC_BATCH_REQUESTS:
        raise AssertionError("anthropic request count mismatch")
    if len(set(anthropic_ids)) != len(anthropic_ids):
        raise AssertionError("anthropic duplicate custom_id")
    for cid in anthropic_ids:
        if not ANTHROPIC_CUSTOM_ID_RE.match(cid):
            raise AssertionError(f"anthropic custom_id fails regex: {cid!r}")
    print(f"Anthropic requests: {len(anthropic_requests)} unique custom_ids, all match ^[a-zA-Z0-9_-]{{1,64}}$")

    qwen_val = _validate_jsonl_lines(qwen_path, QWEN_SYNC_REQUESTS)
    print(f"Qwen worklist: {qwen_val}")

    openai_bytes = openai_path.stat().st_size
    anthropic_bytes = anthropic_path.stat().st_size
    qwen_bytes = qwen_path.stat().st_size
    print(f"File sizes: OpenAI {openai_bytes/1e6:.2f} MB, Anthropic {anthropic_bytes/1e6:.2f} MB, Qwen {qwen_bytes/1e6:.2f} MB")

    if len(openai_cells) > OPENAI_MAX_REQUESTS:
        raise AssertionError("OpenAI exceeds max requests per batch")
    if openai_bytes > OPENAI_MAX_BYTES:
        raise AssertionError("OpenAI batch file exceeds 200 MB — split required")
    if len(anthropic_requests) > ANTHROPIC_MAX_REQUESTS:
        raise AssertionError("Anthropic exceeds max requests per batch")
    if anthropic_bytes > ANTHROPIC_MAX_BYTES:
        raise AssertionError("Anthropic batch payload exceeds 256 MB — split required")
    print("Provider limits: PASS (single batch each, under count and size caps)")

    _validate_all_fallback(plan, openai_cells, anthropic_cells, qwen_cells)
    print(f"All-fallback {ALL_FALLBACK_ENTRY_ID}: absent from (a), present in (b)/(c) — PASS")

    max_cid_len = max(len(c.custom_id) for c in plan)
    print(f"custom_id max length: {max_cid_len} (Anthropic limit 64, OpenAI limit 512)")

    # Cost estimate omitted from public release.

    print("\n=== Doc URLs used ===")
    for key, url in DOC_URLS.items():
        print(f"  {key}: {url}")

    print("\n=== STOP — Phase 1 complete. No submission. Awaiting explicit go-ahead. ===")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
