#!/usr/bin/env python3
"""Phase 2 — submit OpenAI + Anthropic batches and start Qwen sync.

Run from cats-converter/ (requires API keys in repo-root .env):

    uv run python ../eval/part2/full_run_phase2.py --run-dir full_eval_20260618T053204Z

Options:
    --submit-only     Submit batches only; do not start Qwen sync
    --qwen-only       Resume/start Qwen sync only (skip batch submit)
    --qwen-sync       Run Qwen to completion in foreground (default: background)

Phase 3/4 are separate scripts (not invoked here).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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

from eval.part2.batch_submit import submit_anthropic_batch, submit_openai_batch  # noqa: E402
from eval.part2.qwen_sync_run import run_qwen_sync  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_run_dir(name: str) -> Path:
    path = RUNS_ROOT / name
    if not path.is_dir():
        raise SystemExit(f"run dir not found: {path}")
    manifest_path = path / "phase1_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing phase1_manifest.json in {path}")
    return path


def _load_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "phase1_manifest.json").read_text(encoding="utf-8"))


def _write_submission_record(run_dir: Path, record: dict) -> Path:
    path = run_dir / "phase2_submission.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def submit_batches(run_dir: Path, manifest: dict) -> dict:
    openai_key = os.environ.get("OPENAI_API_KEY_PART2")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY_PART2")
    if not openai_key:
        raise SystemExit("OPENAI_API_KEY_PART2 not set")
    if not anthropic_key:
        raise SystemExit("ANTHROPIC_API_KEY_PART2")

    paths = manifest["paths"]
    run_id = manifest["run_id"]
    metadata_tag = manifest["metadata_tag"]

    print(f"Submitting OpenAI batch ({manifest['counts']['openai_batch_lines']} requests)…")
    openai_record = submit_openai_batch(
        api_key=openai_key,
        jsonl_path=Path(paths["openai_batch_input"]),
        run_id=run_id,
        metadata_tag=metadata_tag,
    )
    print(f"  batch_id: {openai_record['batch_id']}")
    print(f"  status:   {openai_record['status']}")

    print(f"Submitting Anthropic batch ({manifest['counts']['anthropic_batch_requests']} requests)…")
    anthropic_record = submit_anthropic_batch(
        api_key=anthropic_key,
        requests_path=Path(paths["anthropic_batch_requests"]),
        run_id=run_id,
        metadata_tag=metadata_tag,
    )
    print(f"  batch_id: {anthropic_record['batch_id']}")
    print(f"  status:   {anthropic_record['processing_status']}")

    submission = {
        "phase": 2,
        "submitted_at": _utc_now(),
        "run_id": run_id,
        "metadata_tag": metadata_tag,
        "custom_id_sidecar": paths["custom_id_sidecar"],
        "openai": openai_record,
        "anthropic": anthropic_record,
    }
    out_path = _write_submission_record(run_dir, submission)
    print(f"\nSubmission record: {out_path}")
    return submission


def start_qwen_sync(
    run_dir: Path,
    manifest: dict,
    *,
    foreground: bool,
) -> Path:
    openrouter_key = os.environ.get("OPENROUTER_API_KEY_PART2")
    if not openrouter_key:
        raise SystemExit("OPENROUTER_API_KEY_PART2 not set")

    worklist = Path(manifest["paths"]["qwen_sync_worklist"])
    checkpoint = run_dir / "qwen_sync_checkpoint.jsonl"
    run_id = manifest["run_id"]

    if foreground:
        run_qwen_sync(
            worklist_path=worklist,
            checkpoint_path=checkpoint,
            api_key=openrouter_key,
            run_id=run_id,
        )
        return checkpoint

    log_path = run_dir / "qwen_sync_runner.log"
    cmd = [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "eval" / "part2" / "full_run_phase2.py"),
        "--run-dir",
        run_dir.name,
        "--qwen-only",
        "--qwen-sync",
    ]
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(CATS_CONVERTER),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )
    print(f"Qwen sync started in background (pid={proc.pid})")
    print(f"  checkpoint: {checkpoint}")
    print(f"  log:        {log_path}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 2 full eval — Phase 2 submit")
    parser.add_argument(
        "--run-dir",
        default="full_eval_20260618T053204Z",
        help="Run directory name under eval/part2/runs/",
    )
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument("--qwen-only", action="store_true")
    parser.add_argument(
        "--qwen-sync",
        action="store_true",
        help="Run Qwen sequentially in foreground (used by background launcher)",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    manifest = _load_manifest(run_dir)

    if args.qwen_only:
        if not args.qwen_sync:
            start_qwen_sync(run_dir, manifest, foreground=False)
            return
        openrouter_key = os.environ.get("OPENROUTER_API_KEY_PART2")
        if not openrouter_key:
            raise SystemExit("OPENROUTER_API_KEY_PART2 not set")
        checkpoint = run_dir / "qwen_sync_checkpoint.jsonl"
        run_qwen_sync(
            worklist_path=Path(manifest["paths"]["qwen_sync_worklist"]),
            checkpoint_path=checkpoint,
            api_key=openrouter_key,
            run_id=manifest["run_id"],
        )
        return

    submission_path = run_dir / "phase2_submission.json"
    if submission_path.exists():
        print(f"WARNING: {submission_path} already exists — skipping batch submit")
        submission = json.loads(submission_path.read_text(encoding="utf-8"))
        print(f"  OpenAI batch_id:    {submission['openai']['batch_id']}")
        print(f"  Anthropic batch_id: {submission['anthropic']['batch_id']}")
    else:
        submission = submit_batches(run_dir, manifest)

    if args.submit_only:
        print("\n--submit-only: Qwen sync not started")
        return

    start_qwen_sync(run_dir, manifest, foreground=args.qwen_sync)


if __name__ == "__main__":
    main()
