#!/usr/bin/env python3
"""Coverage-side charts for Part 1 records.jsonl:

  1. Fallback-cause breakdown (horizontal bar chart) -- what the residual
     non-coverage actually is.
  2. Per-category per-tool savings (box plot), averaged across ALL populated
     tokenizers -- a tokenizer-agnostic view of whether the headline reduction
     is uniform across BFCL categories.

Each tool's savings = mean of its per-tokenizer percent reductions (tiktoken,
qwen, anthropic) over whichever tokenizers are populated for that tool. Also
prints a per-category median table.

Run::

    uv run --project cats-converter python eval/part1/coverage_charts.py eval/out/records.jsonl
    # optional args: [records.jsonl] [output_dir]
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TOKENIZERS = ("tiktoken", "qwen", "anthropic")


def _mean_reduction_across_tokenizers(row) -> float | None:
    """Mean of per-tokenizer percent reductions over populated tokenizers."""
    reds = []
    for tok in TOKENIZERS:
        j = row.get(f"tokens_json_{tok}")
        c = row.get(f"tokens_cats_{tok}")
        if j is not None and c is not None and j > 0:
            reds.append((j - c) / j * 100)
    return statistics.mean(reds) if reds else None


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/out/records.jsonl")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

    # ---- 1. Fallback-cause bar chart ----
    causes = Counter(
        r["fallback_reason"]
        for r in rows
        if r["bucket"] == "fell_back" and r.get("fallback_reason")
    )
    if causes:
        labels, counts = zip(*sorted(causes.items(), key=lambda kv: kv[1]))
        fig, ax = plt.subplots(figsize=(10, max(2.5, 0.5 * len(labels) + 1.5)))
        ax.barh(range(len(labels)), counts, color="tab:red", alpha=0.75)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([l if len(l) <= 60 else l[:57] + "..." for l in labels],
                           fontsize=8)
        ax.set_xlabel("number of tools (unique defs)")
        ax.set_title(f"Fallback causes (total {sum(counts)} fell back)")
        for i, c in enumerate(counts):
            ax.annotate(str(c), (c, i), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        p = out_dir / "fallback_causes.png"
        fig.savefig(p, dpi=150)
        print(f"Wrote: {p}  ({len(labels)} distinct causes)")
    else:
        print("No fallbacks -- no fallback-cause chart.")

    # ---- 2. Per-category savings box plot (averaged across tokenizers) ----
    # report how many tokenizers were actually averaged, for honesty
    tok_present = Counter()
    by_cat = defaultdict(list)
    for r in rows:
        if r["bucket"] != "converted":
            continue
        red = _mean_reduction_across_tokenizers(r)
        if red is None:
            continue
        for tok in TOKENIZERS:
            if r.get(f"tokens_json_{tok}") is not None:
                tok_present[tok] += 1
        for cat in r.get("source_categories", ["(unknown)"]):
            by_cat[cat].append(red)

    if not by_cat:
        print("No converted token data -- no per-category chart.")
        return

    averaged = [t for t in TOKENIZERS if tok_present[t]]
    print(f"\nAveraging across tokenizers present: {', '.join(averaged)}")
    print("Per-category per-tool % reduction (mean across tokenizers):")
    print(f"  {'category':24} {'n':>5} {'median':>8} {'mean':>8}")
    cats_sorted = sorted(by_cat.items(),
                         key=lambda kv: statistics.median(kv[1]), reverse=True)
    for cat, vals in cats_sorted:
        print(f"  {cat:24} {len(vals):5d} {statistics.median(vals):7.2f}% "
              f"{statistics.mean(vals):7.2f}%")

    labels = [c for c, _ in cats_sorted]
    data = [v for _, v in cats_sorted]
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(labels) + 2), 6))
    ax.boxplot(data, showmeans=True,
               meanprops={"marker": "D", "markerfacecolor": "black",
                          "markeredgecolor": "black", "markersize": 5},
               medianprops={"color": "tab:blue", "linewidth": 2},
               flierprops={"marker": ".", "markersize": 3, "alpha": 0.3})
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([f"{l}\n(n={len(d)})" for l, d in zip(labels, data)],
                       rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Per-tool token reduction (%) -- mean across tokenizers")
    ax.set_title(f"CATS savings by BFCL category (avg of {', '.join(averaged)})")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "savings_by_category_avg.png"
    fig.savefig(p, dpi=150)
    print(f"\nWrote: {p}")


if __name__ == "__main__":
    main()