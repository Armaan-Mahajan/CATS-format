"""Fetch batch results and build the immutable raw store for Part 2 eval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from eval.part2.checker import PART2_CLAUDE_MODEL, PART2_OPENAI_MODEL, PART2_QWEN_MODEL


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class BatchPollStatus:
    provider: str
    batch_id: str
    ready: bool
    status: str
    detail: dict[str, Any]


def poll_openai_batch(api_key: str, batch_id: str) -> BatchPollStatus:
    from openai import OpenAI

    batch = OpenAI(api_key=api_key).batches.retrieve(batch_id)
    payload = batch.model_dump(mode="json")
    status = batch.status
    ready = status in {"completed", "expired"}
    return BatchPollStatus(
        provider="openai",
        batch_id=batch_id,
        ready=ready,
        status=status,
        detail=payload,
    )


def poll_anthropic_batch(api_key: str, batch_id: str) -> BatchPollStatus:
    from anthropic import Anthropic

    batch = Anthropic(api_key=api_key).messages.batches.retrieve(batch_id)
    payload = batch.model_dump(mode="json")
    status = batch.processing_status
    ready = status == "ended"
    return BatchPollStatus(
        provider="anthropic",
        batch_id=batch_id,
        ready=ready,
        status=status,
        detail=payload,
    )


def _write_once(path: Path, text: str) -> bool:
    """Write file only if it does not exist. Returns True if written."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def fetch_openai_batch_files(
    *,
    api_key: str,
    batch_id: str,
    out_dir: Path,
) -> dict[str, Path | None]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    batch = client.batches.retrieve(batch_id)
    if batch.status not in {"completed", "expired"}:
        raise RuntimeError(f"OpenAI batch {batch_id} not fetchable (status={batch.status})")

    paths: dict[str, Path | None] = {"output": None, "error": None}
    out_dir.mkdir(parents=True, exist_ok=True)
    if batch.output_file_id:
        output_path = out_dir / f"openai_{batch_id}_output.jsonl"
        if not output_path.exists():
            output_path.write_text(client.files.content(batch.output_file_id).text, encoding="utf-8")
        paths["output"] = output_path

    if batch.error_file_id:
        error_path = out_dir / f"openai_{batch_id}_error.jsonl"
        if not error_path.exists():
            error_path.write_text(client.files.content(batch.error_file_id).text, encoding="utf-8")
        paths["error"] = error_path

    if paths["output"] is None and paths["error"] is None:
        raise RuntimeError(f"OpenAI batch {batch_id} has no output or error file")

    return paths


def fetch_anthropic_batch_file(
    *,
    api_key: str,
    batch_id: str,
    out_dir: Path,
) -> Path:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        raise RuntimeError(
            f"Anthropic batch {batch_id} not fetchable (processing_status={batch.processing_status})"
        )

    out_path = out_dir / f"anthropic_{batch_id}_output.jsonl"
    if out_path.exists():
        return out_path

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for result in client.messages.batches.results(batch_id):
            handle.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return out_path


def _parse_openai_results(output_path: Path | None, error_path: Path | None) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}

    def _consume(path: Path | None) -> None:
        if path is None or not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            custom_id = row["custom_id"]
            top_error = row.get("error")
            response = row.get("response") or {}
            status_code = response.get("status_code")
            if top_error is not None:
                by_id[custom_id] = {
                    "status": "api_error",
                    "raw": row,
                    "error": top_error,
                }
            elif status_code == 200:
                by_id[custom_id] = {
                    "status": "completed",
                    "raw": response.get("body"),
                    "error": None,
                }
            else:
                by_id[custom_id] = {
                    "status": "api_error",
                    "raw": row,
                    "error": response.get("body") or row.get("error") or f"HTTP {status_code}",
                }

    _consume(output_path)
    _consume(error_path)
    return by_id


def _parse_anthropic_results(output_path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        custom_id = row["custom_id"]
        result = row.get("result") or {}
        result_type = result.get("type")
        if result_type == "succeeded":
            message = result.get("message")
            by_id[custom_id] = {
                "status": "completed",
                "raw": message,
                "error": None,
            }
        else:
            by_id[custom_id] = {
                "status": "api_error",
                "raw": row,
                "error": result_type or "unknown_batch_result",
            }
    return by_id


def _parse_qwen_checkpoint(checkpoint_path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        custom_id = row["custom_id"]
        if row.get("status") == "completed" and row.get("http_status") == 200:
            by_id[custom_id] = {
                "status": "completed",
                "raw": row.get("response_body"),
                "error": row.get("extraction_error"),
            }
        else:
            by_id[custom_id] = {
                "status": "api_error",
                "raw": row.get("response_body"),
                "error": row.get("error") or row.get("extraction_error") or "qwen_sync_error",
            }
    return by_id


def _provider_for_model(model: str) -> str:
    if model == PART2_OPENAI_MODEL:
        return "openai"
    if model == PART2_CLAUDE_MODEL:
        return "anthropic"
    if model == PART2_QWEN_MODEL:
        return "qwen"
    raise ValueError(f"unknown model: {model}")


def build_raw_store_records(
    *,
    sidecar_cells: dict[str, dict[str, Any]],
    openai_by_id: dict[str, dict[str, Any]],
    anthropic_by_id: dict[str, dict[str, Any]],
    qwen_by_id: dict[str, dict[str, Any]],
    run_id: str,
    raw_refs: dict[str, str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for custom_id in sorted(sidecar_cells.keys()):
        meta = sidecar_cells[custom_id]
        if meta.get("skipped"):
            records.append(
                {
                    "custom_id": custom_id,
                    "run_id": run_id,
                    "entry_id": meta["entry_id"],
                    "category": meta["category"],
                    "condition": meta["condition"],
                    "model": meta["model"],
                    "provider": _provider_for_model(str(meta["model"])),
                    "status": "skipped",
                    "skip_reason": meta.get("skip_reason"),
                    "raw_response_ref": None,
                    "raw_response": None,
                    "error": None,
                }
            )
            continue

        provider = _provider_for_model(str(meta["model"]))
        if provider == "openai":
            result = openai_by_id.get(custom_id)
            ref_key = "openai_output"
        elif provider == "anthropic":
            result = anthropic_by_id.get(custom_id)
            ref_key = "anthropic_output"
        else:
            result = qwen_by_id.get(custom_id)
            ref_key = "qwen_checkpoint"

        if result is None:
            records.append(
                {
                    "custom_id": custom_id,
                    "run_id": run_id,
                    "entry_id": meta["entry_id"],
                    "category": meta["category"],
                    "condition": meta["condition"],
                    "model": meta["model"],
                    "provider": provider,
                    "status": "api_error",
                    "skip_reason": None,
                    "raw_response_ref": raw_refs.get(ref_key),
                    "raw_response": None,
                    "error": "missing_from_provider_results",
                }
            )
            continue

        records.append(
            {
                "custom_id": custom_id,
                "run_id": run_id,
                "entry_id": meta["entry_id"],
                "category": meta["category"],
                "condition": meta["condition"],
                "model": meta["model"],
                "provider": provider,
                "status": result["status"],
                "skip_reason": None,
                "raw_response_ref": raw_refs.get(ref_key),
                "raw_response": result["raw"],
                "error": result.get("error"),
            }
        )
    return records


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for record in records:
        provider = record["provider"]
        bucket = summary.setdefault(
            provider,
            {"completed": 0, "api_error": 0, "skipped": 0, "api_error_custom_ids": []},
        )
        status = record["status"]
        bucket[status] = bucket.get(status, 0) + 1
        if status == "api_error":
            bucket["api_error_custom_ids"].append(record["custom_id"])
    for bucket in summary.values():
        bucket["api_error_custom_ids"].sort()
    return summary
