"""Submit full Part 2 eval batches (OpenAI + Anthropic)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.part2.batch_io import OPENAI_BATCH_ENDPOINT
from eval.part2.checker import PART2_CLAUDE_MODEL, PART2_OPENAI_MODEL


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def submit_openai_batch(
    *,
    api_key: str,
    jsonl_path: Path,
    run_id: str,
    metadata_tag: str,
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    jsonl_text = jsonl_path.read_text(encoding="utf-8")
    line_count = sum(1 for line in jsonl_text.splitlines() if line.strip())

    with jsonl_path.open("rb") as handle:
        batch_file = client.files.create(file=handle, purpose="batch")

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint=OPENAI_BATCH_ENDPOINT,
        completion_window="24h",
        metadata={
            "purpose": metadata_tag,
            "run_id": run_id,
            "model": PART2_OPENAI_MODEL,
        },
    )

    return {
        "provider": "openai",
        "batch_id": batch.id,
        "input_file_id": batch_file.id,
        "endpoint": OPENAI_BATCH_ENDPOINT,
        "model": PART2_OPENAI_MODEL,
        "status": batch.status,
        "request_count": line_count,
        "input_jsonl_path": str(jsonl_path),
        "submitted_at": _utc_now(),
        "raw_batch": batch.model_dump(mode="json"),
    }


def submit_anthropic_batch(
    *,
    api_key: str,
    requests_path: Path,
    run_id: str,
    metadata_tag: str,
) -> dict[str, Any]:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    payload = json.loads(requests_path.read_text(encoding="utf-8"))
    requests = payload["requests"]

    batch = client.messages.batches.create(requests=requests)

    return {
        "provider": "anthropic",
        "batch_id": batch.id,
        "endpoint": "/v1/messages/batches",
        "model": PART2_CLAUDE_MODEL,
        "processing_status": batch.processing_status,
        "request_count": len(requests),
        "requests_path": str(requests_path),
        "submitted_at": _utc_now(),
        "metadata_tag": metadata_tag,
        "run_id": run_id,
        "raw_batch": batch.model_dump(mode="json"),
    }
