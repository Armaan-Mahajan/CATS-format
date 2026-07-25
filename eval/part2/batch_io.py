"""Batch file construction — wraps bodies from build_request(); no body logic here."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.part2.cell_plan import CellSpec
from eval.part2.request_builder import PreparedRequest, build_request

# Doc-verified limits (Phase 1 research, June 2026):
# OpenAI Batch: ≤50,000 requests, ≤200 MB file, custom_id ≤512 chars
#   https://platform.openai.com/docs/api-reference/batch/create
#   https://platform.openai.com/docs/api-reference/batch/request-input
# Anthropic Message Batches: ≤100,000 requests OR ≤256 MB body, custom_id ≤64 chars
#   ^[a-zA-Z0-9_-]{1,64}$
#   https://docs.anthropic.com/en/api/creating-message-batches

OPENAI_BATCH_ENDPOINT = "/v1/responses"
OPENAI_BATCH_METHOD = "POST"
OPENAI_MAX_REQUESTS = 50_000
OPENAI_MAX_BYTES = 200 * 1024 * 1024
ANTHROPIC_MAX_REQUESTS = 100_000
ANTHROPIC_MAX_BYTES = 256 * 1024 * 1024


def prepared_request_for_cell(cell: CellSpec, entry_by_id: dict) -> PreparedRequest:
    return build_request(
        entry_by_id[cell.entry_id],
        model=cell.model,
        condition=cell.condition,
    )


def openai_batch_line(custom_id: str, body: dict[str, Any]) -> str:
    return json.dumps(
        {
            "custom_id": custom_id,
            "method": OPENAI_BATCH_METHOD,
            "url": OPENAI_BATCH_ENDPOINT,
            "body": body,
        },
        ensure_ascii=False,
    )


def anthropic_batch_request(custom_id: str, body: dict[str, Any]) -> dict[str, Any]:
    return {"custom_id": custom_id, "params": body}


def build_openai_batch_jsonl(cells: list[CellSpec], entry_by_id: dict) -> tuple[str, list[dict[str, Any]]]:
    lines: list[str] = []
    meta: list[dict[str, Any]] = []
    for cell in cells:
        prepared = prepared_request_for_cell(cell, entry_by_id)
        if prepared.skipped or prepared.request_body is None:
            raise ValueError(f"unexpected skip in OpenAI batch cell {cell.custom_id}")
        lines.append(openai_batch_line(cell.custom_id, prepared.request_body))
        meta.append({"custom_id": cell.custom_id, "entry_id": cell.entry_id, "condition": cell.condition.value})
    return "\n".join(lines) + ("\n" if lines else ""), meta


def build_anthropic_batch_requests(cells: list[CellSpec], entry_by_id: dict) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    for cell in cells:
        prepared = prepared_request_for_cell(cell, entry_by_id)
        if prepared.skipped or prepared.request_body is None:
            raise ValueError(f"unexpected skip in Anthropic batch cell {cell.custom_id}")
        requests.append(anthropic_batch_request(cell.custom_id, prepared.request_body))
        meta.append({"custom_id": cell.custom_id, "entry_id": cell.entry_id, "condition": cell.condition.value})
    return requests, meta


def build_qwen_worklist(cells: list[CellSpec], entry_by_id: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        prepared = prepared_request_for_cell(cell, entry_by_id)
        if prepared.skipped or prepared.request_body is None:
            raise ValueError(f"unexpected skip in Qwen worklist cell {cell.custom_id}")
        rows.append(
            {
                "custom_id": cell.custom_id,
                "entry_id": cell.entry_id,
                "category": cell.category,
                "condition": cell.condition.value,
                "request_body": prepared.request_body,
            }
        )
    return rows


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
