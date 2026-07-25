"""Sequential, resumable Qwen/Parasail sync run for full Part 2 eval."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from eval.part2.checker import Part2Condition
from eval.part2.http_client import http_json_with_retry
from eval.part2.live_client import _openrouter_backend
from eval.part2.providers import Provider
from eval.part2.response_parser import extract_model_output

_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/Armaan-Mahajan/CATS-format",
    "X-OpenRouter-Title": "CATS Part 2 Eval",
}

EXPECTED_BACKEND = "parasail"
DEFAULT_MAX_429_RETRIES = 8
DEFAULT_INITIAL_BACKOFF_S = 2.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_worklist(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_checkpoint_custom_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("custom_id"):
            ids.add(row["custom_id"])
    return ids


def _append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _call_openrouter(
    *,
    api_key: str,
    body: dict[str, Any],
    condition: Part2Condition,
    max_retries: int,
) -> dict[str, Any]:
    """One OpenRouter call with 429 backoff; fail loudly after cap."""
    attempt = 0
    backoff = DEFAULT_INITIAL_BACKOFF_S
    last_status = 0
    last_body: Any = None

    while attempt <= max_retries:
        status, response = http_json_with_retry(
            method="POST",
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                **_OPENROUTER_HEADERS,
            },
            body=body,
            max_retries=0,
            initial_backoff_s=backoff,
        )
        last_status = status
        last_body = response

        if status == 429 and attempt < max_retries:
            time.sleep(backoff)
            backoff *= 2
            attempt += 1
            continue

        if status != 200:
            return {
                "http_status": status,
                "response_body": response if isinstance(response, (dict, list, str)) else None,
                "backend": None,
                "extracted": None,
                "error": f"HTTP {status}",
                "retry_count": attempt,
                "status": "api_error",
            }

        if not isinstance(response, dict):
            return {
                "http_status": status,
                "response_body": None,
                "backend": None,
                "extracted": None,
                "error": "non-dict response",
                "retry_count": attempt,
                "status": "api_error",
            }

        gen_id = response.get("id")
        backend = _openrouter_backend(
            response,
            str(gen_id) if gen_id else None,
            api_key,
        )
        if backend is None or backend.lower() != EXPECTED_BACKEND:
            return {
                "http_status": status,
                "response_body": response,
                "backend": backend,
                "extracted": None,
                "error": f"backend_assertion_failed: expected Parasail, got {backend!r}",
                "retry_count": attempt,
                "status": "api_error",
            }

        try:
            extracted = extract_model_output(
                provider=Provider.OPENROUTER,
                condition=condition,
                response=response,
            )
            extracted_payload: dict[str, Any] | None = {
                "prompt_text": extracted.prompt_text,
                "model_output": extracted.model_output,
                "parse_outcome": extracted.parse_outcome.value if extracted.parse_outcome else None,
                "response_tool_name": extracted.response_tool_name,
            }
            extraction_error = None
        except Exception as exc:  # noqa: BLE001 — bad model output must not kill the run
            extracted_payload = None
            extraction_error = f"{type(exc).__name__}: {exc}"

        return {
            "http_status": status,
            "response_body": response,
            "backend": backend,
            "extracted": extracted_payload,
            "extraction_error": extraction_error,
            "error": None,
            "retry_count": attempt,
            "status": "completed",
        }

    return {
        "http_status": last_status,
        "response_body": last_body if isinstance(last_body, (dict, list, str)) else None,
        "backend": None,
        "extracted": None,
        "error": f"HTTP 429 after {max_retries} retries",
        "retry_count": attempt,
        "status": "api_error",
    }


def iter_qwen_sync(
    *,
    worklist_path: Path,
    checkpoint_path: Path,
    api_key: str,
    run_id: str,
    max_429_retries: int = DEFAULT_MAX_429_RETRIES,
    progress_every: int = 25,
) -> Iterator[dict[str, Any]]:
    """Yield checkpoint records; append each to checkpoint_path."""
    worklist = _load_worklist(worklist_path)
    done = _load_checkpoint_custom_ids(checkpoint_path)
    pending = [row for row in worklist if row["custom_id"] not in done]

    yield {
        "event": "start",
        "run_id": run_id,
        "worklist_total": len(worklist),
        "already_done": len(done),
        "pending": len(pending),
        "checkpoint_path": str(checkpoint_path),
        "timestamp": _utc_now(),
    }

    for index, row in enumerate(pending, start=1):
        condition = Part2Condition(row["condition"])
        try:
            result = _call_openrouter(
                api_key=api_key,
                body=row["request_body"],
                condition=condition,
                max_retries=max_429_retries,
            )
        except Exception as exc:  # noqa: BLE001 — never abort a 2954-cell run mid-loop
            result = {
                "http_status": 0,
                "response_body": None,
                "backend": None,
                "extracted": None,
                "extraction_error": None,
                "error": f"{type(exc).__name__}: {exc}",
                "retry_count": 0,
                "status": "api_error",
            }
        record = {
            "custom_id": row["custom_id"],
            "entry_id": row["entry_id"],
            "condition": row["condition"],
            "run_id": run_id,
            "timestamp": _utc_now(),
            **result,
        }
        _append_checkpoint(checkpoint_path, record)
        yield record

        if progress_every and index % progress_every == 0:
            yield {
                "event": "progress",
                "completed_this_run": index,
                "pending_remaining": len(pending) - index,
                "last_custom_id": row["custom_id"],
                "timestamp": _utc_now(),
            }

    yield {
        "event": "done",
        "run_id": run_id,
        "checkpoint_path": str(checkpoint_path),
        "total_in_checkpoint": len(_load_checkpoint_custom_ids(checkpoint_path)),
        "timestamp": _utc_now(),
    }


def run_qwen_sync(
    *,
    worklist_path: Path,
    checkpoint_path: Path,
    api_key: str,
    run_id: str,
    max_429_retries: int = DEFAULT_MAX_429_RETRIES,
) -> dict[str, Any]:
    """Run full sync loop; print progress to stdout."""
    stats = {"completed": 0, "api_error": 0}
    for event in iter_qwen_sync(
        worklist_path=worklist_path,
        checkpoint_path=checkpoint_path,
        api_key=api_key,
        run_id=run_id,
        max_429_retries=max_429_retries,
    ):
        if event.get("event") == "start":
            print(
                f"Qwen sync: {event['pending']} pending "
                f"({event['already_done']} already checkpointed) "
                f"-> {event['checkpoint_path']}",
                flush=True,
            )
        elif event.get("event") == "progress":
            print(
                f"  progress: {event['completed_this_run']} this run, "
                f"{event['pending_remaining']} remaining "
                f"(last {event['last_custom_id']})",
                flush=True,
            )
        elif event.get("event") == "done":
            print(
                f"Qwen sync complete: {event['total_in_checkpoint']} rows in checkpoint",
                flush=True,
            )
            return {
                "checkpoint_path": str(checkpoint_path),
                "total_in_checkpoint": event["total_in_checkpoint"],
                **stats,
            }
        elif "custom_id" in event:
            status = event.get("status", "api_error")
            stats[status] = stats.get(status, 0) + 1
            if status == "api_error":
                print(
                    f"  API ERROR {event['custom_id']}: {event.get('error')}",
                    file=sys.stderr,
                    flush=True,
                )
            elif event.get("extraction_error"):
                print(
                    f"  extraction warning {event['custom_id']}: {event['extraction_error']}",
                    file=sys.stderr,
                    flush=True,
                )
    return {"checkpoint_path": str(checkpoint_path), **stats}
