#!/usr/bin/env python3
"""Part 2 inferential statistics over scored_cells.jsonl.

Run from the repository root:

    uv run --project cats-converter python eval/part2/compute_stats.py

The margin, the discordant-count threshold, and the CI methods are pre-specified
constants defined below; see the paper's Statistical Method section for the
rationale. Outputs under results/part2/raw/.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

REPO_ROOT = Path(__file__).resolve().parents[2]
CATS_CONVERTER = REPO_ROOT / "cats-converter"
STATS_DIR = REPO_ROOT / "results" / "part2" / "raw"
SCORED_PATH = REPO_ROOT / "results" / "part2" / "raw" / "scored_cells.jsonl"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CATS_CONVERTER) not in sys.path:
    sys.path.insert(0, str(CATS_CONVERTER))

from eval.part2.checker import (  # noqa: E402
    PART2_CLAUDE_MODEL,
    PART2_OPENAI_MODEL,
    PART2_QWEN_MODEL,
)
from eval.part2.pilot_entries import PILOT_ENTRY_ALL_FALLBACK  # noqa: E402

MODELS = [PART2_OPENAI_MODEL, PART2_CLAUDE_MODEL, PART2_QWEN_MODEL]
MODEL_SHORT = {
    PART2_OPENAI_MODEL: "gpt",
    PART2_CLAUDE_MODEL: "claude",
    PART2_QWEN_MODEL: "qwen",
}
CONDITIONS = ["cats_in_prompt", "json_in_prompt", "native_tools"]
CONDITION_SHORT = {
    "cats_in_prompt": "a_cats",
    "json_in_prompt": "b_json",
    "native_tools": "c_native",
}

# Pre-specified, before the real p-values were computed. Not tuned to the results.
DELTA = 0.03
ALPHA = 0.05
DISCORDANT_EXACT_THRESHOLD = 25  # use exact McNemar below this discordant count

# Anthropic api_error — provider schema limitation, not model/CATS failure.
EXCLUDED_CUSTOM_IDS = frozenset({"c0006749"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = proportion_confint(k, n, alpha=ALPHA, method="wilson")
    return k / n, float(lo), float(hi)


def _is_usable(cell: dict[str, Any]) -> bool:
    if cell["custom_id"] in EXCLUDED_CUSTOM_IDS:
        return False
    if cell["status"] == "skipped":
        return False
    if cell["status"] == "api_error":
        return False
    return cell["status"] == "completed"


def load_scored(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index_cells(cells: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(c["model"], c["entry_id"], c["condition"]): c for c in cells}


@dataclass(frozen=True)
class Paired2x2:
    model: str
    comparison: str
    metric: str
    n: int
    both_success: int
    a_only: int  # first condition success, second failure
    b_only: int  # first failure, second success
    both_fail: int

    @property
    def discordant(self) -> int:
        return self.a_only + self.b_only


def build_paired_2x2(
    *,
    indexed: dict[tuple[str, str, str], dict[str, Any]],
    model: str,
    cond_a: str,
    cond_b: str,
    comparison: str,
    metric: str,
) -> Paired2x2:
    """Pair by entry where both conditions are usable; metric is 'syntactic', 'semantic', or 'joint'.

    'semantic' (the pre-specified primary) conditions on syntactic validity: entries
    where either arm failed to parse are dropped. Because the notation itself affects
    parse rates, that conditions the estimand on a post-treatment outcome (a potential
    collider). 'joint' is the unconditional robustness companion: every paired entry
    counts, and a syntactic failure scores as an overall failure
    (success = syntactic_valid AND semantic_correct) instead of being excluded.
    """
    entries = set()
    for key in indexed:
        if key[0] == model:
            entries.add(key[1])

    n11 = n10 = n01 = n00 = 0
    for entry_id in sorted(entries):
        if entry_id == PILOT_ENTRY_ALL_FALLBACK and (
            cond_a == "cats_in_prompt" or cond_b == "cats_in_prompt"
        ):
            continue
        ca = indexed.get((model, entry_id, cond_a))
        cb = indexed.get((model, entry_id, cond_b))
        if ca is None or cb is None:
            continue
        if not _is_usable(ca) or not _is_usable(cb):
            continue

        if metric == "syntactic":
            sa = bool(ca["syntactic_valid"])
            sb = bool(cb["syntactic_valid"])
        elif metric == "semantic":
            if not ca["syntactic_valid"] or not cb["syntactic_valid"]:
                continue
            sa = ca["semantic_correct"] is True
            sb = cb["semantic_correct"] is True
        elif metric == "joint":
            sa = bool(ca["syntactic_valid"]) and ca["semantic_correct"] is True
            sb = bool(cb["syntactic_valid"]) and cb["semantic_correct"] is True
        else:
            raise ValueError(metric)

        if sa and sb:
            n11 += 1
        elif sa and not sb:
            n10 += 1
        elif not sa and sb:
            n01 += 1
        else:
            n00 += 1

    return Paired2x2(
        model=model,
        comparison=comparison,
        metric=metric,
        n=n11 + n10 + n01 + n00,
        both_success=n11,
        a_only=n10,
        b_only=n01,
        both_fail=n00,
    )


def mcnemar_test(table: Paired2x2) -> dict[str, Any]:
    """McNemar on discordant pairs; table rows = cond_a outcome, cols = cond_b."""
    if table.n == 0:
        return {
            "mcnemar_statistic": None,
            "mcnemar_pvalue": None,
            "mcnemar_variant": "n/a",
        }

    arr = np.array(
        [
            [table.both_success, table.a_only],
            [table.b_only, table.both_fail],
        ]
    )
    use_exact = table.discordant < DISCORDANT_EXACT_THRESHOLD
    result = mcnemar(arr, exact=use_exact)
    variant = (
        "exact_binomial (discordant < 25)"
        if use_exact
        else "continuity_corrected_chi_square"
    )
    return {
        "mcnemar_statistic": float(result.statistic) if result.statistic is not None else None,
        "mcnemar_pvalue": float(result.pvalue),
        "mcnemar_variant": variant,
        "discordant_count": table.discordant,
    }


def paired_difference_ci(table: Paired2x2) -> dict[str, Any]:
    """95% CI for (cond_a − cond_b) proportion difference via Agresti–Min (2005) paired Wald."""
    n = table.n
    if n == 0:
        return {
            "difference_a_minus_b": float("nan"),
            "difference_ci_lower": float("nan"),
            "difference_ci_upper": float("nan"),
            "difference_ci_method": "Agresti–Min (2005) paired Wald on McNemar table",
        }

    diff = (table.a_only - table.b_only) / n
    # Agresti & Min (2005), Statistics in Medicine — variance for paired proportion difference.
    numer = table.a_only + table.b_only - (table.a_only - table.b_only) ** 2 / n
    numer = max(numer, 0.0)
    se = np.sqrt(numer) / n
    z = 1.959963984540054  # 95% two-sided
    return {
        "difference_a_minus_b": diff,
        "difference_ci_lower": diff - z * se,
        "difference_ci_upper": diff + z * se,
        "difference_ci_method": "Agresti–Min (2005) paired Wald SE on McNemar table",
    }


def non_inferiority_verdict(ci_lower: float, ci_upper: float) -> str:
    """δ = 3% one-sided non-inferiority on CATS−JSON semantic difference."""
    if ci_lower > -DELTA:
        return "pass"
    if ci_upper <= -DELTA:
        return "fail"
    return "inconclusive"


def descriptive_table(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for condition in CONDITIONS:
            subset = [
                c
                for c in cells
                if c["model"] == model
                and c["condition"] == condition
                and _is_usable(c)
            ]
            n = len(subset)
            syn_k = sum(1 for c in subset if c["syntactic_valid"])
            syn_rate, syn_lo, syn_hi = _wilson(syn_k, n)
            sem_subset = [c for c in subset if c["syntactic_valid"]]
            sem_n = len(sem_subset)
            sem_k = sum(1 for c in sem_subset if c["semantic_correct"] is True)
            sem_rate, sem_lo, sem_hi = _wilson(sem_k, sem_n)
            rows.append(
                {
                    "model": model,
                    "model_short": MODEL_SHORT[model],
                    "condition": condition,
                    "condition_short": CONDITION_SHORT[condition],
                    "n_completed": n,
                    "syntactic_valid_count": syn_k,
                    "syntactic_valid_rate": syn_rate,
                    "syntactic_valid_ci_lower": syn_lo,
                    "syntactic_valid_ci_upper": syn_hi,
                    "semantic_n_syntactically_valid": sem_n,
                    "semantic_correct_count": sem_k,
                    "semantic_correct_rate": sem_rate,
                    "semantic_correct_ci_lower": sem_lo,
                    "semantic_correct_ci_upper": sem_hi,
                }
            )
    return rows


def paired_row(table: Paired2x2, *, headline: bool = False) -> dict[str, Any]:
    mcn = mcnemar_test(table)
    diff = paired_difference_ci(table)
    row = {
        "model": table.model,
        "model_short": MODEL_SHORT[table.model],
        "comparison": table.comparison,
        "metric": table.metric,
        "paired_n": table.n,
        "both_success": table.both_success,
        "a_only_success": table.a_only,
        "b_only_success": table.b_only,
        "both_fail": table.both_fail,
        **mcn,
        **diff,
    }
    if headline and table.metric == "semantic" and table.comparison == "a_cats_vs_b_json":
        row["delta_non_inferiority"] = DELTA
        row["non_inferiority_verdict"] = non_inferiority_verdict(
            diff["difference_ci_lower"], diff["difference_ci_upper"]
        )
    return row


def audit_reconciliation(indexed: dict) -> dict[str, Any]:
    """Compare (a) vs (b) semantic discordant counts to pre-audit preview."""
    preview = {
        PART2_OPENAI_MODEL: {"cats_wrong_json_right": 25, "cats_right_json_wrong": 19},
        PART2_CLAUDE_MODEL: {"cats_wrong_json_right": 45, "cats_right_json_wrong": 27},
        PART2_QWEN_MODEL: {"cats_wrong_json_right": 49, "cats_right_json_wrong": 24},
    }
    out: dict[str, Any] = {}
    for model, prev in preview.items():
        table = build_paired_2x2(
            indexed=indexed,
            model=model,
            cond_a="cats_in_prompt",
            cond_b="json_in_prompt",
            comparison="a_cats_vs_b_json",
            metric="semantic",
        )
        actual = {
            "cats_wrong_json_right": table.b_only,  # a✗ b✓
            "cats_right_json_wrong": table.a_only,  # a✓ b✗
            "paired_n": table.n,
        }
        match = actual["cats_wrong_json_right"] == prev["cats_wrong_json_right"] and (
            actual["cats_right_json_wrong"] == prev["cats_right_json_wrong"]
        )
        out[MODEL_SHORT[model]] = {
            "preview": prev,
            "actual_with_exclusions": actual,
            "match_preview": match,
        }
    return out


def write_stats_summary(
    path: Path,
    descriptive: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    reconciliation: dict[str, Any],
) -> None:
    headline = [r for r in paired if r["comparison"] == "a_cats_vs_b_json" and r["metric"] == "semantic"]
    lines = [
        "Part 2 inferential statistics summary",
        f"Generated: {_utc_now()}",
        f"Input: {SCORED_PATH}",
        f"Pre-specified non-inferiority margin delta = {DELTA:.0%}",
        "",
        "Exclusions: 3 skipped all-fallback (a) cells; Anthropic api_error c0006749.",
        "",
        "=== Headline: CATS (a) vs JSON (b) semantic correctness, per model ===",
        "",
    ]
    for row in headline:
        lines.extend(
            [
                f"Model: {row['model_short']} ({row['model']})",
                f"  Paired n (both syntactically valid, non-excluded): {row['paired_n']}",
                f"  2x2: both_ok={row['both_success']}  a_ok_b_fail={row['a_only_success']}  "
                f"a_fail_b_ok={row['b_only_success']}  both_fail={row['both_fail']}",
                f"  CATS - JSON difference: {row['difference_a_minus_b']:.4f} "
                f"(95% CI [{row['difference_ci_lower']:.4f}, {row['difference_ci_upper']:.4f}]; "
                f"{row['difference_ci_method']})",
                f"  McNemar p = {row['mcnemar_pvalue']:.6f} ({row['mcnemar_variant']})",
                f"  Non-inferiority vs delta=3%: {row['non_inferiority_verdict']} "
                f"(CI lower bound = {row['difference_ci_lower']:.4f})",
                "",
            ]
        )

    verdicts = {r["model_short"]: r["non_inferiority_verdict"] for r in headline}
    lines.append(
        "Three-model pattern (factual): "
        f"GPT={verdicts.get('gpt')}, Claude={verdicts.get('claude')}, Qwen={verdicts.get('qwen')}."
    )
    lines.append("")

    lines.append("=== Audit reconciliation (preview vs paired semantic a vs b) ===")
    for model_key, rec in reconciliation.items():
        lines.append(
            f"  {model_key}: match={rec['match_preview']}  preview={rec['preview']}  "
            f"actual={rec['actual_with_exclusions']}"
        )
    lines.append("")

    lines.append("=== Descriptive Wilson CIs (see descriptive_rates.csv) ===")
    for row in descriptive:
        lines.append(
            f"  {row['model_short']} {row['condition_short']}: "
            f"syn {row['syntactic_valid_rate']:.3f} [{row['syntactic_valid_ci_lower']:.3f},"
            f"{row['syntactic_valid_ci_upper']:.3f}] n={row['n_completed']}; "
            f"sem {row['semantic_correct_rate']:.3f} [{row['semantic_correct_ci_lower']:.3f},"
            f"{row['semantic_correct_ci_upper']:.3f}] n={row['semantic_n_syntactically_valid']}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cells = load_scored(SCORED_PATH)
    indexed = index_cells(cells)

    descriptive = descriptive_table(cells)

    paired_rows: list[dict[str, Any]] = []
    for model in MODELS:
        # Headline + secondary semantic comparisons
        for cond_a, cond_b, label in (
            ("cats_in_prompt", "json_in_prompt", "a_cats_vs_b_json"),
            ("cats_in_prompt", "native_tools", "a_cats_vs_c_native"),
            ("json_in_prompt", "native_tools", "b_json_vs_c_native"),
        ):
            table = build_paired_2x2(
                indexed=indexed,
                model=model,
                cond_a=cond_a,
                cond_b=cond_b,
                comparison=label,
                metric="semantic",
            )
            paired_rows.append(
                paired_row(
                    table,
                    headline=(label == "a_cats_vs_b_json"),
                )
            )
        # Syntactic paired (a) vs (b) only
        table_syn = build_paired_2x2(
            indexed=indexed,
            model=model,
            cond_a="cats_in_prompt",
            cond_b="json_in_prompt",
            comparison="a_cats_vs_b_json",
            metric="syntactic",
        )
        paired_rows.append(paired_row(table_syn, headline=False))

    reconciliation = audit_reconciliation(indexed)

    manifest = {
        "generated_at": _utc_now(),
        "input_path": str(SCORED_PATH),
        "delta_non_inferiority": DELTA,
        "exclusions": {
            "skipped_all_fallback_entry": PILOT_ENTRY_ALL_FALLBACK,
            "skipped_count": 3,
            "api_error_custom_ids": sorted(EXCLUDED_CUSTOM_IDS),
        },
        "descriptive": descriptive,
        "paired_comparisons": paired_rows,
        "audit_reconciliation": reconciliation,
        "mcnemar_rule": (
            f"exact binomial when discordant < {DISCORDANT_EXACT_THRESHOLD}, "
            "else continuity-corrected chi-square (statsmodels.stats.contingency_tables.mcnemar)"
        ),
        "difference_ci_method": "Agresti–Min (2005) paired Wald SE on McNemar table",
        "wilson_method": "statsmodels.stats.proportion.proportion_confint(method='wilson')",
    }

    json_path = STATS_DIR / "part2_stats_results.json"
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = STATS_DIR / "part2_stats_results.csv"
    if paired_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0].keys()))
            writer.writeheader()
            writer.writerows(paired_rows)

    desc_csv = STATS_DIR / "descriptive_rates.csv"
    with desc_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(descriptive[0].keys()))
        writer.writeheader()
        writer.writerows(descriptive)

    summary_path = STATS_DIR / "stats_summary.txt"
    write_stats_summary(summary_path, descriptive, paired_rows, reconciliation)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {desc_csv}")
    print(f"Wrote {summary_path}")
    print()
    for row in paired_rows:
        if row["comparison"] == "a_cats_vs_b_json" and row["metric"] == "semantic":
            print(
                f"{row['model_short']}: n={row['paired_n']} "
                f"diff={row['difference_a_minus_b']:.4f} "
                f"CI=[{row['difference_ci_lower']:.4f},{row['difference_ci_upper']:.4f}] "
                f"p={row['mcnemar_pvalue']:.4f} "
                f"NI={row['non_inferiority_verdict']}"
            )


if __name__ == "__main__":
    main()
