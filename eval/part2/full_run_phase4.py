#!/usr/bin/env python3
"""Phase 4 — score all cells from raw_store.jsonl (re-runnable).

Run from cats-converter/:

    uv run python ../eval/part2/full_run_phase4.py --run-dir full_eval_20260618T053204Z

Outputs:
  scored_cells.jsonl
  scored_cells.csv
  phase4_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CATS_CONVERTER = REPO_ROOT / "cats-converter"
RUNS_ROOT = REPO_ROOT / "eval" / "part2" / "runs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CATS_CONVERTER) not in sys.path:
    sys.path.insert(0, str(CATS_CONVERTER))

from eval.part2.score_raw_store import (  # noqa: E402
    STATS_TASK_NOTE,
    ScoredCell,
    score_raw_record,
    summarize_scored_cells,
    write_csv,
    _utc_now,
)


def _resolve_run_dir(name: str) -> Path:
    path = RUNS_ROOT / name
    if not path.is_dir():
        raise SystemExit(f"run dir not found: {path}")
    return path


def run_phase4(run_dir: Path, *, force: bool = False) -> dict:
    raw_store = run_dir / "raw_store.jsonl"
    if not raw_store.exists():
        raise SystemExit(f"missing raw_store: {raw_store}")

    out_jsonl = run_dir / "scored_cells.jsonl"
    out_csv = run_dir / "scored_cells.csv"
    if out_jsonl.exists() and not force:
        print(f"scored_cells.jsonl already exists — loading ({out_jsonl})")
        cells = [
            ScoredCell(**{k: row[k] for k in (
                "custom_id", "entry_id", "category", "model", "condition", "provider",
                "status", "syntactic_valid", "semantic_correct", "parse_outcome",
                "raw_response_ref", "scored_via", "skip_reason", "api_error",
            )})
            for line in out_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in [json.loads(line)]
        ]
    else:
        from eval.part2.corpus import load_part2_corpus  # noqa: E402

        entry_by_id = {entry.id: entry for entry in load_part2_corpus()}
        cells: list[ScoredCell] = []
        for line in raw_store.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            entry = entry_by_id.get(record["entry_id"])
            if entry is None:
                raise SystemExit(f"corpus entry missing: {record['entry_id']}")
            cells.append(score_raw_record(record, entry))

        rows = [cell.as_dict() for cell in cells]
        with out_jsonl.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_csv(out_csv, rows)
        print(f"Wrote {len(rows)} rows -> {out_jsonl}")
        print(f"Wrote CSV -> {out_csv}")

    status_counts: dict[str, int] = {}
    for cell in cells:
        status_counts[cell.status] = status_counts.get(cell.status, 0) + 1

    summary = summarize_scored_cells(cells)
    report = {
        "phase": 4,
        "scored_at": _utc_now(),
        "run_id": run_dir.name,
        "raw_store_path": str(raw_store),
        "scored_cells_jsonl": str(out_jsonl),
        "scored_cells_csv": str(out_csv),
        "cell_count": len(cells),
        "status_counts": status_counts,
        "model_condition_summary": summary,
        "stats_task_note": STATS_TASK_NOTE,
    }
    report_path = run_dir / "phase4_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _print_summary(report: dict) -> None:
    print("\n=== Per model × condition (completed cells only) ===")
    for key in sorted(report["model_condition_summary"]):
        bucket = report["model_condition_summary"][key]
        print(
            f"  {bucket['model']} | {bucket['condition']}: "
            f"n={bucket['n']} "
            f"syn={bucket['syntactic_valid_count']} ({bucket['syntactic_valid_rate']:.1%}) "
            f"sem={bucket['semantic_correct_count']} ({bucket['semantic_correct_rate']:.1%} of syntactic-valid)"
        )
    print(f"\nStatus totals: {report['status_counts']}")
    print(f"\nStats task note: {report['stats_task_note']}")
    print(f"\nReport: {report['report_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 2 full eval — Phase 4 score")
    parser.add_argument("--run-dir", default="full_eval_20260618T053204Z")
    parser.add_argument("--force", action="store_true", help="Re-score even if scored_cells.jsonl exists")
    args = parser.parse_args()

    report = run_phase4(_resolve_run_dir(args.run_dir), force=args.force)
    _print_summary(report)


if __name__ == "__main__":
    main()
