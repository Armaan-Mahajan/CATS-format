#!/usr/bin/env python3
"""Phase 3 — poll batch jobs, fetch results, build immutable raw store.

Run from cats-converter/:

    uv run python ../eval/part2/full_run_phase3.py --run-dir full_eval_20260618T053204Z

    uv run python ../eval/part2/full_run_phase3.py --run-dir full_eval_20260618T053204Z --poll-only

Re-runnable: poll anytime; fetch skips existing raw_fetch files and won't overwrite raw_store.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
CATS_CONVERTER = REPO_ROOT / "cats-converter"
RUNS_ROOT = REPO_ROOT / "eval" / "part2" / "runs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CATS_CONVERTER) not in sys.path:
    sys.path.insert(0, str(CATS_CONVERTER))

load_dotenv(REPO_ROOT / ".env")

from eval.part2.batch_fetch import (  # noqa: E402
    build_raw_store_records,
    fetch_anthropic_batch_file,
    fetch_openai_batch_files,
    poll_anthropic_batch,
    poll_openai_batch,
    summarize_records,
    _parse_anthropic_results,
    _parse_openai_results,
    _parse_qwen_checkpoint,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_run_dir(name: str) -> Path:
    path = RUNS_ROOT / name
    if not path.is_dir():
        raise SystemExit(f"run dir not found: {path}")
    return path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def poll_batches(run_dir: Path) -> dict[str, dict]:
    submission = _load_json(run_dir / "phase2_submission.json")
    openai_key = os.environ.get("OPENAI_API_KEY_PART2")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY_PART2")
    if not openai_key or not anthropic_key:
        raise SystemExit("OPENAI_API_KEY_PART2 and ANTHROPIC_API_KEY_PART2 required")

    openai = poll_openai_batch(openai_key, submission["openai"]["batch_id"])
    anthropic = poll_anthropic_batch(anthropic_key, submission["anthropic"]["batch_id"])

    report = {
        "polled_at": _utc_now(),
        "openai": {
            "batch_id": openai.batch_id,
            "status": openai.status,
            "ready_to_fetch": openai.ready,
            "request_counts": openai.detail.get("request_counts"),
        },
        "anthropic": {
            "batch_id": anthropic.batch_id,
            "status": anthropic.status,
            "ready_to_fetch": anthropic.ready,
            "request_counts": anthropic.detail.get("request_counts"),
        },
    }
    poll_path = run_dir / "phase3_poll.json"
    poll_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def fetch_and_build_store(run_dir: Path) -> dict:
    submission = _load_json(run_dir / "phase2_submission.json")
    manifest = _load_json(run_dir / "phase1_manifest.json")
    sidecar = _load_json(Path(manifest["paths"]["custom_id_sidecar"]))
    run_id = submission["run_id"]

    openai_key = os.environ.get("OPENAI_API_KEY_PART2")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY_PART2")
    if not openai_key or not anthropic_key:
        raise SystemExit("OPENAI_API_KEY_PART2 and ANTHROPIC_API_KEY_PART2 required")

    raw_fetch_dir = run_dir / "raw_fetch"
    raw_store_path = run_dir / "raw_store.jsonl"

    if raw_store_path.exists():
        print(f"raw_store already exists: {raw_store_path}")
        records = [
            json.loads(line)
            for line in raw_store_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary = summarize_records(records)
        return {
            "run_id": run_id,
            "raw_store_path": str(raw_store_path),
            "record_count": len(records),
            "summary": summary,
            "reused_existing": True,
        }

    poll = poll_batches(run_dir)
    if not poll["openai"]["ready_to_fetch"]:
        raise SystemExit(f"OpenAI batch not ready: {poll['openai']['status']}")
    if not poll["anthropic"]["ready_to_fetch"]:
        raise SystemExit(f"Anthropic batch not ready: {poll['anthropic']['status']}")

    openai_paths = fetch_openai_batch_files(
        api_key=openai_key,
        batch_id=submission["openai"]["batch_id"],
        out_dir=raw_fetch_dir,
    )
    anthropic_path = fetch_anthropic_batch_file(
        api_key=anthropic_key,
        batch_id=submission["anthropic"]["batch_id"],
        out_dir=raw_fetch_dir,
    )

    qwen_checkpoint = run_dir / "qwen_sync_checkpoint.jsonl"
    if not qwen_checkpoint.exists():
        raise SystemExit(f"missing Qwen checkpoint: {qwen_checkpoint}")

    openai_by_id = _parse_openai_results(openai_paths["output"], openai_paths["error"])
    anthropic_by_id = _parse_anthropic_results(anthropic_path)
    qwen_by_id = _parse_qwen_checkpoint(qwen_checkpoint)

    raw_refs = {
        "openai_output": str(openai_paths["output"]) if openai_paths["output"] else None,
        "openai_error": str(openai_paths["error"]) if openai_paths["error"] else None,
        "anthropic_output": str(anthropic_path),
        "qwen_checkpoint": str(qwen_checkpoint),
    }

    records = build_raw_store_records(
        sidecar_cells=sidecar["cells"],
        openai_by_id=openai_by_id,
        anthropic_by_id=anthropic_by_id,
        qwen_by_id=qwen_by_id,
        run_id=run_id,
        raw_refs=raw_refs,
    )

    with raw_store_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize_records(records)
    report = {
        "phase": 3,
        "built_at": _utc_now(),
        "run_id": run_id,
        "raw_store_path": str(raw_store_path),
        "raw_fetch_dir": str(raw_fetch_dir),
        "raw_refs": raw_refs,
        "record_count": len(records),
        "summary": summary,
    }
    report_path = run_dir / "phase3_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(records)} records -> {raw_store_path}")
    print(f"Report -> {report_path}")
    return report


def _print_summary(summary: dict) -> None:
    print("\n=== Per-provider counts ===")
    for provider in ("openai", "anthropic", "qwen"):
        bucket = summary.get(provider, {})
        print(
            f"  {provider:10} completed={bucket.get('completed', 0)} "
            f"api_error={bucket.get('api_error', 0)} skipped={bucket.get('skipped', 0)}"
        )
        ids = bucket.get("api_error_custom_ids") or []
        if ids:
            print(f"    api_error custom_ids ({len(ids)}): {', '.join(ids[:10])}" + (
                f" … +{len(ids)-10} more" if len(ids) > 10 else ""
            ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 2 full eval — Phase 3 fetch")
    parser.add_argument("--run-dir", default="full_eval_20260618T053204Z")
    parser.add_argument("--poll-only", action="store_true")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)

    if args.poll_only:
        poll = poll_batches(run_dir)
        print(json.dumps(poll, indent=2))
        return

    report = fetch_and_build_store(run_dir)
    _print_summary(report["summary"])


if __name__ == "__main__":
    main()
