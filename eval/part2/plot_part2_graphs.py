#!/usr/bin/env python3
"""Generate Part 2 accuracy figures from results/part2/raw/.

Run from repo root::

    uv run --project cats-converter python eval/part2/plot_part2_graphs.py

Writes PNGs to results/part2/graphs/.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

# Journal figure style (two-column A4, 170 mm text width, sans-serif body).
matplotlib.rcParams.update({
    "figure.figsize": (6.7, 2.8),   # 170mm text width; keep height tight
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,             # embed Type 1/TrueType, never Type 3
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "results" / "part2" / "raw"
GRAPH_DIR = REPO_ROOT / "results" / "part2" / "graphs"
FIGURES_DIR = REPO_ROOT / "figures"


def _save_paper_figure(fig, name: str) -> None:
    """Dual-save PDF (paper) + 300 dpi PNG (repo) from the same figure object.

    An explicit full-figure bbox pins the page to exactly 6.7 in wide so the
    PDF drops into ``\\linewidth`` without scaling.
    """
    full_bbox = Bbox([[0.0, 0.0], list(fig.get_size_inches())])
    for out, kw in ((FIGURES_DIR / f"{name}.pdf", {}), (GRAPH_DIR / f"{name}.png", {"dpi": 300})):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches=full_bbox, **kw)
        print(f"Wrote {out}")

MODEL_ORDER = ["gpt", "claude", "qwen"]
MODEL_LABELS = {
    "gpt": "GPT-5.4",
    "claude": "Claude Sonnet 4.6",
    "qwen": "Qwen3.5-35B-A3B",
}
CONDITION_ORDER = ["a_cats", "b_json", "c_native"]
CONDITION_LABELS = {
    "a_cats": "CATS (a)",
    "b_json": "JSON (b)",
    "c_native": "Native (c)",
}
CONDITION_COLORS = {
    "a_cats": "#4C78A8",
    "b_json": "#F58518",
    "c_native": "#54A24B",
}

NI_VERDICT_COLORS = {
    "pass": "#54A24B",
    "inconclusive": "#EECA3B",
    "fail": "#E45756",
}


def _read_descriptive(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_paired(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _index_descriptive(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["model_short"], r["condition_short"]): r for r in rows}


def _rate_ci(row: dict, prefix: str) -> tuple[float, float, float]:
    rate = float(row[f"{prefix}_rate"])
    lo = float(row[f"{prefix}_ci_lower"])
    hi = float(row[f"{prefix}_ci_upper"])
    return rate, lo, hi


def plot_condition_facets(
    desc_index: dict,
    *,
    metric_prefix: str,
    ylabel: str,
    title: str,
    outfile: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    x = np.arange(len(CONDITION_ORDER))
    width = 0.55

    for ax, model in zip(axes, MODEL_ORDER):
        rates, yerr_lo, yerr_hi = [], [], []
        for cond in CONDITION_ORDER:
            row = desc_index[(model, cond)]
            rate, lo, hi = _rate_ci(row, metric_prefix)
            rates.append(rate * 100)
            yerr_lo.append(rate * 100 - lo * 100)
            yerr_hi.append(hi * 100 - rate * 100)
        colors = [CONDITION_COLORS[c] for c in CONDITION_ORDER]
        ax.bar(
            x,
            rates,
            width,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            yerr=[yerr_lo, yerr_hi],
            capsize=4,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=9)
        ax.set_title(MODEL_LABELS[model], fontsize=11)
        ax.set_ylim(70, 102)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(100, color="#ccc", linewidth=0.8, zorder=0)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    out = GRAPH_DIR / outfile
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_non_inferiority_forest(paired_rows: list[dict]) -> None:
    rows = [
        r
        for r in paired_rows
        if r["comparison"] == "a_cats_vs_b_json" and r["metric"] == "semantic"
    ]
    rows.sort(key=lambda r: MODEL_ORDER.index(r["model_short"]))

    fig, ax = plt.subplots(figsize=(6.7, 2.6))
    y = np.arange(len(rows))

    lows, highs = [], []
    for i, row in enumerate(rows):
        diff = float(row["difference_a_minus_b"]) * 100
        lo = float(row["difference_ci_lower"]) * 100
        hi = float(row["difference_ci_upper"]) * 100
        lows.append(lo)
        highs.append(hi)
        verdict = row.get("non_inferiority_verdict") or "inconclusive"
        color = NI_VERDICT_COLORS.get(verdict, "#666666")
        ax.plot([lo, hi], [i, i], color=color, linewidth=2, solid_capstyle="round")
        ax.scatter([diff], [i], color=color, s=30, zorder=3)
        ax.annotate(
            f"{verdict}  ({diff:+.1f} pp)",
            (hi, i),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=color,
        )

    ax.axvline(0, color="#888888", linestyle="-", linewidth=1)
    ax.axvline(-3.0, color="#E45756", linestyle="--", linewidth=1.2)
    ax.annotate(
        "−δ = −3 pp",
        (-3.0, 0.96),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        va="top",
        fontsize=8,
        color="#E45756",
    )
    # Right headroom so verdict labels stay inside the axes.
    ax.set_xlim(min(lows + [-3.0]) - 0.5, max(highs) + 2.4)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS[r["model_short"]] for r in rows])
    ax.set_xlabel("CATS − JSON semantic accuracy (percentage points, 95% CI)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(pad=0.4)
    _save_paper_figure(fig, "non_inferiority_forest")
    plt.close(fig)


def plot_mcnemar_stacked(paired_rows: list[dict]) -> None:
    rows = [
        r
        for r in paired_rows
        if r["comparison"] == "a_cats_vs_b_json" and r["metric"] == "semantic"
    ]
    rows.sort(key=lambda r: MODEL_ORDER.index(r["model_short"]))

    labels = ["Both correct", "CATS only", "JSON only", "Both wrong"]
    keys = ["both_success", "a_only_success", "b_only_success", "both_fail"]
    colors = ["#54A24B", "#4C78A8", "#F58518", "#B279A2"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))

    for key, label, color in zip(keys, labels, colors):
        vals = np.array([int(r[key]) for r in rows], dtype=float)
        ax.bar(x, vals, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.6)
        for i, (val, b) in enumerate(zip(vals, bottom)):
            if val > 0:
                ax.text(
                    i,
                    b + val / 2,
                    str(int(val)),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if val > 30 else "black",
                )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{MODEL_LABELS[r['model_short']]}\n(n={r['paired_n']})" for r in rows]
    )
    ax.set_ylabel("Entry count (paired, both syntactically valid)")
    ax.set_title("McNemar table: CATS (a) vs JSON (b) semantic correctness")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = GRAPH_DIR / "mcnemar_a_vs_b_semantic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_prompt_vs_native(paired_rows: list[dict]) -> None:
    # Grayscale-safe: the two series differ by hatch as well as color.
    series = [
        ("a_cats_vs_c_native", "CATS − native (a−c)", "#4C78A8", None),
        ("b_json_vs_c_native", "JSON − native (b−c)", "#F58518", "///"),
    ]
    fig, ax = plt.subplots(figsize=(6.7, 2.8))
    x = np.arange(len(MODEL_ORDER))
    width = 0.32
    offsets = [-width / 2, width / 2]

    for offset, (comparison, label, color, hatch) in zip(offsets, series):
        diffs, err_lo, err_hi, ns = [], [], [], []
        for model in MODEL_ORDER:
            row = next(
                r
                for r in paired_rows
                if r["model_short"] == model
                and r["comparison"] == comparison
                and r["metric"] == "semantic"
            )
            diff = float(row["difference_a_minus_b"]) * 100
            lo = float(row["difference_ci_lower"]) * 100
            hi = float(row["difference_ci_upper"]) * 100
            diffs.append(diff)
            err_lo.append(diff - lo)
            err_hi.append(hi - diff)
            ns.append(int(row["paired_n"]))
        ax.bar(
            x + offset,
            diffs,
            width,
            label=label,
            color=color,
            edgecolor="white",
            hatch=hatch,
            linewidth=0.6,
            yerr=[err_lo, err_hi],
            capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )

    ax.axhline(0, color="#888888", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{MODEL_LABELS[m]}\n(n≈{next(r['paired_n'] for r in paired_rows if r['model_short']==m and r['comparison']=='a_cats_vs_c_native')})"
         for m in MODEL_ORDER],
        fontsize=8,
    )
    ax.set_ylabel("Semantic accuracy difference\n(pp, 95% CI)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(pad=0.4)
    _save_paper_figure(fig, "prompt_vs_native_paired")
    plt.close(fig)


def main() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    desc = _read_descriptive(RAW_DIR / "descriptive_rates.csv")
    paired = _read_paired(RAW_DIR / "part2_stats_results.csv")
    desc_index = _index_descriptive(desc)

    plot_condition_facets(
        desc_index,
        metric_prefix="semantic_correct",
        ylabel="Semantic correct rate (%)",
        title="Semantic correctness given syntactically valid output (95% Wilson CI)",
        outfile="semantic_accuracy_by_condition.png",
    )
    plot_condition_facets(
        desc_index,
        metric_prefix="syntactic_valid",
        ylabel="Syntactic valid rate (%)",
        title="Syntactic validity over completed cells (95% Wilson CI)",
        outfile="syntactic_validity_by_condition.png",
    )
    plot_non_inferiority_forest(paired)
    plot_mcnemar_stacked(paired)
    plot_prompt_vs_native(paired)

    manifest = {
        "graphs": [
            "semantic_accuracy_by_condition.png",
            "syntactic_validity_by_condition.png",
            "non_inferiority_forest.png",
            "mcnemar_a_vs_b_semantic.png",
            "prompt_vs_native_paired.png",
        ],
        "source": "results/part2/raw/descriptive_rates.csv, part2_stats_results.csv",
    }
    (GRAPH_DIR / "graphs_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {GRAPH_DIR / 'graphs_manifest.json'}")


if __name__ == "__main__":
    main()
