#!/usr/bin/env python3
"""Post-hoc joint-success robustness check for the Part 2 headline comparison.

WHY THIS EXISTS
---------------
The pre-specified primary analysis (compute_stats.py, metric="semantic") compares
CATS (a) vs JSON (b) semantic accuracy CONDITIONAL on syntactic validity: an entry
only counts if both arms produced a parseable call. That is defensible (an unparseable
call has no semantic score), but it conditions the estimand on an outcome the
treatment itself affects — the notation changes parse rates. Conditioning on such a
post-treatment variable is a potential collider: if CATS's syntactic failures land
disproportionately on entries that would also have failed semantically, dropping them
could shrink the apparent CATS−JSON semantic gap relative to the true end-to-end gap.

This script re-runs the identical paired machinery (McNemar test, Agresti–Min CI,
delta=3% non-inferiority verdict — all imported from compute_stats.py, not
reimplemented) on the UNCONDITIONAL composite outcome

    joint_success = syntactic_valid AND (semantic_correct is True)

so a syntactic failure counts as a failure instead of excluding the entry. Only the
(a) vs (b) comparison is checked, since that is the pre-specified headline. Outputs
go to NEW files under results/part2/raw/ (robustness_joint_success.txt / .csv); the
pre-specified primary outputs are not touched. This is a post-hoc robustness check,
NOT part of the pre-specified primary analysis.

Run from the repo root:

    uv run python eval/part2/compute_robustness.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.part2.compute_stats import (  # noqa: E402
    DELTA,
    MODEL_SHORT,
    MODELS,
    SCORED_PATH,
    _utc_now,
    build_paired_2x2,
    index_cells,
    load_scored,
    non_inferiority_verdict,
    paired_row,
)

OUT_DIR = REPO_ROOT / "results" / "part2" / "raw"
TXT_PATH = OUT_DIR / "robustness_joint_success.txt"
CSV_PATH = OUT_DIR / "robustness_joint_success.csv"


def main() -> None:
    cells = load_scored(SCORED_PATH)
    indexed = index_cells(cells)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    lines = [
        "Part 2 robustness check: joint (unconditional) success, CATS (a) vs JSON (b)",
        f"Generated: {_utc_now()}",
        f"Input: {SCORED_PATH.relative_to(REPO_ROOT)}",
        "",
        "Post-hoc robustness companion to the pre-specified conditional semantic",
        "analysis (stats_summary.txt). Outcome per entry:",
        "  joint_success = syntactic_valid AND (semantic_correct is True)",
        "Syntactic failures count as failures instead of being excluded, removing the",
        "conditioning on a post-treatment outcome (potential collider). Same paired",
        "McNemar / Agresti-Min CI / delta=3% non-inferiority machinery as the primary.",
        "Same exclusions as the primary: all-fallback entry, api_error c0006749,",
        "skipped/non-completed cells.",
        "",
    ]

    for model in MODELS:
        joint_table = build_paired_2x2(
            indexed=indexed,
            model=model,
            cond_a="cats_in_prompt",
            cond_b="json_in_prompt",
            comparison="a_cats_vs_b_json",
            metric="joint",
        )
        row = paired_row(joint_table, headline=False)
        row["delta_non_inferiority"] = DELTA
        row["non_inferiority_verdict"] = non_inferiority_verdict(
            row["difference_ci_lower"], row["difference_ci_upper"]
        )
        rows.append(row)

        # Recompute the headline conditional semantic result (same code path that
        # produced stats_summary.txt) purely for the side-by-side comparison line.
        sem_table = build_paired_2x2(
            indexed=indexed,
            model=model,
            cond_a="cats_in_prompt",
            cond_b="json_in_prompt",
            comparison="a_cats_vs_b_json",
            metric="semantic",
        )
        sem_row = paired_row(sem_table, headline=True)

        lines.extend(
            [
                f"Model: {row['model_short']} ({row['model']})",
                f"  Paired n (unconditional; no validity conditioning): {row['paired_n']}",
                f"  2x2: both_ok={row['both_success']}  a_ok_b_fail={row['a_only_success']}  "
                f"a_fail_b_ok={row['b_only_success']}  both_fail={row['both_fail']}",
                f"  CATS - JSON difference: {row['difference_a_minus_b']:.4f} "
                f"(95% CI [{row['difference_ci_lower']:.4f}, {row['difference_ci_upper']:.4f}]; "
                f"{row['difference_ci_method']})",
                f"  McNemar p = {row['mcnemar_pvalue']:.6f} ({row['mcnemar_variant']})",
                f"  Non-inferiority vs delta=3%: {row['non_inferiority_verdict']} "
                f"(CI lower bound = {row['difference_ci_lower']:.4f})",
                f"  headline (conditional): {sem_row['non_inferiority_verdict']}, "
                f"CI [{sem_row['difference_ci_lower']:.4f}, {sem_row['difference_ci_upper']:.4f}] "
                f"(n={sem_row['paired_n']}) | joint (unconditional): "
                f"{row['non_inferiority_verdict']}, "
                f"CI [{row['difference_ci_lower']:.4f}, {row['difference_ci_upper']:.4f}] "
                f"(n={row['paired_n']})",
                "",
            ]
        )

    TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {TXT_PATH}")
    print(f"Wrote {CSV_PATH}")
    print()
    for row in rows:
        print(
            f"{row['model_short']}: n={row['paired_n']} "
            f"diff={row['difference_a_minus_b']:.4f} "
            f"CI=[{row['difference_ci_lower']:.4f},{row['difference_ci_upper']:.4f}] "
            f"p={row['mcnemar_pvalue']:.4f} "
            f"NI={row['non_inferiority_verdict']}"
        )


if __name__ == "__main__":
    main()
