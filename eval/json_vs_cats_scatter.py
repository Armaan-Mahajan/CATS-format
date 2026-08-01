#!/usr/bin/env python3
"""Scatter: per-tool JSON tokens vs CATS tokens (converted tools only).

Each point is one unique converted tool: x = compact normalized JSON token count,
y = CATS token count. Points below y = x are token wins; vertical gap to the line is
absolute savings. By default all three tokenizers are overlaid (distinct colors).

Run::

    uv run --project cats-converter python eval/json_vs_cats_scatter.py
    uv run --project cats-converter python eval/json_vs_cats_scatter.py eval/out/records.jsonl tiktoken
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECORDS = REPO_ROOT / "eval" / "out" / "records.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "eval" / "out"
FIGURES_DIR = REPO_ROOT / "figures"

TOKENIZERS = ("tiktoken", "qwen", "anthropic")
# Grayscale-safe: series differ by marker as well as color.
TOKENIZER_MARKERS = {"tiktoken": "o", "qwen": "s", "anthropic": "^"}


def _save_paper_figure(fig, name: str, png_dir: Path) -> None:
    """Dual-save PDF (paper) + 300 dpi PNG (repo) from the same figure object.

    An explicit full-figure bbox pins the page to exactly 6.7 in wide so the
    PDF drops into ``\\linewidth`` without scaling.
    """
    full_bbox = Bbox([[0.0, 0.0], list(fig.get_size_inches())])
    for out, kw in ((FIGURES_DIR / f"{name}.pdf", {}), (png_dir / f"{name}.png", {"dpi": 300})):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches=full_bbox, **kw)
        print(f"Wrote {out}")

TOKENIZER_LABELS = {
    "tiktoken": "tiktoken o200k_base — GPT-5.X",
    "qwen": "qwen — Qwen3.5-35B-A3B",
    "anthropic": "anthropic — Claude Sonnet 4.6 / Opus 4.6",
}

TOKENIZER_COLORS = {
    "tiktoken": "tab:blue",
    "qwen": "tab:orange",
    "anthropic": "tab:green",
}


def _load_points(records_path: Path, tok: str) -> list[tuple[str, int, int]]:
    jkey, ckey = f"tokens_json_{tok}", f"tokens_cats_{tok}"
    points: list[tuple[str, int, int]] = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("bucket") != "converted":
            continue
        j = row.get(jkey)
        c = row.get(ckey)
        if j is None or c is None or j <= 0:
            continue
        points.append((row["tool_hash"], int(j), int(c)))
    return points


def _summarize(tok: str, points: list[tuple[str, int, int]]) -> tuple[int, int, list[tuple[str, int, int]]]:
    wins = sum(1 for _, j, c in points if c < j)
    bad = [(h, j, c) for h, j, c in points if c >= j]
    return wins, len(bad), bad


def _print_summary(records_path: Path, tok: str, points: list[tuple[str, int, int]]) -> None:
    n = len(points)
    n_wins, n_bad, regressions = _summarize(tok, points)
    print(f"== {tok} ({TOKENIZER_LABELS[tok]}) ==")
    print(f"  converted tools plotted: {n}")
    print(f"  below y=x (CATS smaller): {n_wins}")
    print(f"  on/above y=x (tie or regression): {n_bad}")
    if regressions:
        print("  tool_hashes with cats_tokens >= json_tokens:")
        for h, j, c in sorted(regressions, key=lambda t: (c - j, t[0]), reverse=True):
            print(f"    {h}  json={j}  cats={c}  delta={c - j:+d}")
    else:
        print("  no regressions — every tool strictly below y=x.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "records",
        nargs="?",
        type=Path,
        default=DEFAULT_RECORDS,
        help=f"path to records.jsonl (default: {DEFAULT_RECORDS})",
    )
    parser.add_argument(
        "tokenizer",
        nargs="?",
        default="all",
        choices=["all", *TOKENIZERS],
        help="tokenizer to plot, or 'all' for overlay (default: all)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output PNG (default: eval/out/json_vs_cats.png or json_vs_cats_<tok>.png)",
    )
    args = parser.parse_args()

    records_path = args.records
    tok_arg = args.tokenizer
    tokenizers = list(TOKENIZERS) if tok_arg == "all" else [tok_arg]

    series: list[tuple[str, list[tuple[str, int, int]]]] = []
    for tok in tokenizers:
        points = _load_points(records_path, tok)
        if not points:
            print(
                f"WARNING: skipping {tok} — no converted tools with token columns "
                f"(did you run --with-anthropic?)",
                file=sys.stderr,
            )
            continue
        series.append((tok, points))

    if not series:
        print(f"No tokenizer data in {records_path}", file=sys.stderr)
        raise SystemExit(1)

    print(f"records: {records_path}\n")
    title_parts: list[str] = []
    all_xs: list[int] = []
    all_ys: list[int] = []

    for tok, points in series:
        _print_summary(records_path, tok, points)
        n_wins, _, _ = _summarize(tok, points)
        title_parts.append(f"{tok} {n_wins}/{len(points)}")
        all_xs.extend(j for _, j, _ in points)
        all_ys.extend(c for _, _, c in points)

    if tok_arg == "all":
        out_path = args.output or (DEFAULT_OUT_DIR / "json_vs_cats.png")
        xlabel = "JSON tokens (compact normalized tool schema)"
        ylabel = "CATS tokens"
        title = "JSON vs CATS per tool — " + " · ".join(title_parts)
    else:
        out_path = args.output or (DEFAULT_OUT_DIR / f"json_vs_cats_{tok_arg}.png")
        xlabel = f"JSON tokens ({tok_arg}, compact normalized tool schema)"
        ylabel = f"CATS tokens ({tok_arg})"
        tok, points = series[0]
        n_wins, _, _ = _summarize(tok, points)
        title = f"{TOKENIZER_LABELS[tok]}: {n_wins}/{len(points)} tools below y=x"

    lo = min(min(all_xs), min(all_ys))
    hi = max(max(all_xs), max(all_ys))
    pad = max(4, int((hi - lo) * 0.04))
    axis_lo = max(0, lo - pad)
    axis_hi = hi + pad

    fig, ax = plt.subplots(figsize=(6.7, 2.8))
    for tok, points in series:
        xs = [j for _, j, _ in points]
        ys = [c for _, _, c in points]
        n_wins, n_bad, _ = _summarize(tok, points)
        # Rasterize only the scatter layer: text/axes stay vector, file stays small.
        ax.scatter(
            xs,
            ys,
            s=6,
            alpha=0.25,
            color=TOKENIZER_COLORS[tok],
            marker=TOKENIZER_MARKERS[tok],
            edgecolors="none",
            rasterized=True,
            label=f"{tok} ({n_wins}/{len(points)} below y=x)",
        )

    ax.plot(
        [axis_lo, axis_hi],
        [axis_lo, axis_hi],
        color="tab:red",
        linewidth=1.2,
        linestyle="--",
        label="y = x (no savings)",
        zorder=5,
    )

    ax.set_xlim(axis_lo, axis_hi)
    ax.set_ylim(axis_lo, axis_hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout(pad=0.4)
    # Inline label on the reference line (above it — all points sit below y=x),
    # rotated to the on-screen slope of the line.
    fig.canvas.draw()
    p1 = ax.transData.transform((axis_lo, axis_lo))
    p2 = ax.transData.transform((axis_hi, axis_hi))
    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    mid = axis_lo + 0.45 * (axis_hi - axis_lo)
    ax.annotate(
        "y = x (no savings)",
        (mid, mid),
        xytext=(0, 5),
        textcoords="offset points",
        rotation=angle,
        rotation_mode="anchor",
        ha="center",
        va="bottom",
        fontsize=8,
        color="tab:red",
    )
    _save_paper_figure(fig, out_path.stem, out_path.parent)
    plt.close(fig)


if __name__ == "__main__":
    main()
