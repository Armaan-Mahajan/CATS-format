#!/usr/bin/env python3
"""Token-reduction statistics + box plots for Part 1 records.jsonl.

For each tokenizer (tiktoken, qwen, anthropic), reports per-tool percent-reduction
and absolute token-savings distributions (min, P10, Q1, median, mean, Q3, P90, max)
over CONVERTED tools, weighting each unique tool equally. Percent reduction also
includes the corpus-wide aggregate (sum cats / sum json).

Also reports a robustness comparison vs pretty-printed normalized JSON
(``json.dumps(..., indent=2, sort_keys=True)``) for tiktoken and qwen by default;
pass ``--with-anthropic-pretty`` to include Anthropic (API-cached separately from
compact counts in ``eval/cache/anthropic_pretty_tokens.json``).

Writes four percent box plots (compact, absolute, pretty-printed, compact-vs-pretty combined)
plus annotations.

Run::

    uv run --project cats-converter python eval/part1/token_stats.py eval/out/records.jsonl
    uv run --project cats-converter python eval/part1/token_stats.py eval/out/records.jsonl \\
        --with-anthropic-pretty
    # optional second positional = compact percent box plot (default: eval/out/reduction_boxplot.png)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Callable, Literal

from dotenv import load_dotenv

load_dotenv()

import matplotlib

matplotlib.use("Agg")  # headless: write file, no display needed
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CATS_CONVERTER = REPO_ROOT / "cats-converter"
_SCRIPT_DIR = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(REPO_ROOT / "eval" / ".hf_cache")
# Running as ``python eval/part1/token_stats.py`` puts this folder on sys.path[0],
# which can shadow stdlib/third-party names (e.g. constants) during HF imports.
while str(_SCRIPT_DIR) in sys.path:
    sys.path.remove(str(_SCRIPT_DIR))
for _entry in (str(REPO_ROOT), str(CATS_CONVERTER)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

TOKENIZERS = ("tiktoken", "qwen", "anthropic")
PRETTY_LOCAL_TOKENIZERS = ("tiktoken", "qwen")

BOX_LABELS = {
    "tiktoken": "tiktoken o200k_base — GPT-5.X",
    "qwen": "qwen — Qwen3.5-35B-A3B",
    "anthropic": "anthropic — Claude Sonnet 4.6 / Opus 4.6",
}

# Box width is 0.45 → right edge at x + 0.225; labels start just beyond that.
LABEL_X_OFFSET = 0.26

MetricKind = Literal["percent", "absolute"]


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolation percentile (matches numpy/pandas default 'linear')."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _token_pairs_for(converted: list[dict], tok: str) -> list[tuple[int, int]]:
    jkey, ckey = f"tokens_json_{tok}", f"tokens_cats_{tok}"
    return [
        (r[jkey], r[ckey])
        for r in converted
        if r.get(jkey) is not None and r.get(ckey) is not None and r[jkey] > 0
    ]


def _reductions_for(converted: list[dict], tok: str) -> list[float]:
    return sorted(((j - c) / j) * 100 for j, c in _token_pairs_for(converted, tok))


def _absolute_deltas_for(converted: list[dict], tok: str) -> list[float]:
    """Per-tool json - cats token counts (positive = CATS smaller)."""
    return sorted(float(j - c) for j, c in _token_pairs_for(converted, tok))


def _aggregate_for(converted: list[dict], tok: str):
    pairs = _token_pairs_for(converted, tok)
    if not pairs:
        return None
    tj = sum(j for j, _ in pairs)
    tc = sum(c for _, c in pairs)
    return (tj - tc) / tj * 100, tj, tc


def _box_summary(values: list[float]) -> dict[str, float]:
    return {
        "min": values[0],
        "q1": _percentile(values, 25),
        "q2": _percentile(values, 50),
        "q3": _percentile(values, 75),
        "max": values[-1],
    }


def _format_metric(value: float, kind: MetricKind) -> str:
    if kind == "percent":
        return f"{value:6.2f}%"
    return f"{value:6.1f}"


def _print_distribution_stats(
    label: str,
    values: list[float],
    *,
    kind: MetricKind,
    series_label: str,
    show_heading: bool = True,
) -> None:
    stats = _box_summary(values)
    mean = statistics.mean(values)
    p10 = _percentile(values, 10)
    p90 = _percentile(values, 90)
    if show_heading:
        print(f"== {label} ==  (n = {len(values)})")
    print(f"  {series_label}")
    for key, val in (
        ("min", stats["min"]),
        ("P10", p10),
        ("Q1", stats["q1"]),
        ("median", stats["q2"]),
        ("mean", mean),
        ("Q3", stats["q3"]),
        ("P90", p90),
        ("max", stats["max"]),
    ):
        print(f"    {key:<7}: {_format_metric(val, kind)}")
    worse = sum(1 for v in values if v < 0)
    if worse:
        print(f"  NOTE: {worse} observation(s) grew under CATS (negative savings).")
    print()


def _print_reduction_stats(label: str, reductions: list[float], *, show_heading: bool = True) -> None:
    _print_distribution_stats(
        label,
        reductions,
        kind="percent",
        series_label="Per-tool % reduction (each observation weighted equally):",
        show_heading=show_heading,
    )


def _print_absolute_stats(label: str, deltas: list[float], *, show_heading: bool = True) -> None:
    _print_distribution_stats(
        label,
        deltas,
        kind="absolute",
        series_label="Per-tool absolute savings in tokens (json - cats; each observation weighted equally):",
        show_heading=show_heading,
    )


def _print_tokenizer_stats(label: str, reductions: list[float], deltas: list[float]) -> None:
    print(f"== {label} ==  (n = {len(reductions)})")
    _print_reduction_stats(label, reductions, show_heading=False)
    _print_absolute_stats(label, deltas, show_heading=False)


def _spread_vertical(ys: list[float], min_sep: float) -> list[float]:
    """Nudge display y-positions apart when labels would overlap."""
    if len(ys) <= 1:
        return list(ys)
    out = list(ys)
    for _ in range(len(out) * 3):
        changed = False
        for i in range(1, len(out)):
            gap = out[i] - out[i - 1]
            if gap < min_sep:
                bump = (min_sep - gap) / 2
                out[i - 1] -= bump
                out[i] += bump
                changed = True
        if not changed:
            break
    return out


def _annotate_box(ax, x: int, values: list[float], *, kind: MetricKind) -> None:
    stats = _box_summary(values)
    mean = statistics.mean(values)
    p10 = _percentile(values, 10)
    p90 = _percentile(values, 90)

    entries = [
        (stats["max"], "max", "black"),
        (p90, "P90", "tab:green"),
        (stats["q3"], "Q3", "black"),
        (mean, "mean", "black"),
        (stats["q2"], "median", "tab:blue"),
        (stats["q1"], "Q1", "black"),
        (p10, "P10", "tab:red"),
        (stats["min"], "min", "black"),
    ]
    entries.sort(key=lambda item: item[0])
    min_sep = 1.5 if kind == "percent" else max(2.0, (stats["max"] - stats["min"]) * 0.04)
    display_ys = _spread_vertical([y for y, _, _ in entries], min_sep=min_sep)

    x_text = x + LABEL_X_OFFSET
    for (y, label, color), display_y in zip(entries, display_ys):
        suffix = "%" if kind == "percent" else " tok"
        fmt = f"{y:.1f}{suffix}" if kind == "percent" else f"{y:.0f}{suffix}"
        ax.annotate(
            f"{label} {fmt}",
            (x_text, display_y),
            fontsize=7,
            va="center",
            ha="left",
            color=color,
        )
    ax.plot([x], [p10], marker="_", color="tab:red", markersize=12)
    ax.plot([x], [p90], marker="_", color="tab:green", markersize=12)


def _write_boxplot(
    box_data: list[list[float]],
    box_labels: list[str],
    out_img: Path,
    *,
    kind: MetricKind,
    title: str | None = None,
    ylabel: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(3.4 * len(box_data) + 2, 6.5))
    ax.boxplot(
        box_data,
        widths=0.45,
        showmeans=True,
        meanline=False,
        meanprops={
            "marker": "D",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 6,
        },
        medianprops={"color": "tab:blue", "linewidth": 2},
        flierprops={"marker": ".", "markersize": 4, "alpha": 0.4},
    )
    ax.set_xticks(range(1, len(box_labels) + 1))
    ax.set_xticklabels(box_labels, fontsize=8)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    elif kind == "percent":
        ax.set_ylabel("Per-tool token reduction (%)  --  higher = CATS smaller")
    else:
        ax.set_ylabel("Per-tool absolute token savings (json - cats)  --  higher = CATS smaller")
    if title is not None:
        ax.set_title(title)
    elif kind == "percent":
        ax.set_title("CATS per-tool token reduction by tokenizer")
    else:
        ax.set_title("CATS per-tool absolute token savings by tokenizer")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(0.55, len(box_data) + 1.0)

    for i, values in enumerate(box_data, start=1):
        _annotate_box(ax, i, values, kind=kind)

    fig.tight_layout()
    out_img.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_img, dpi=150)
    plt.close(fig)
    print(f"Wrote box plot: {out_img}")


def _ensure_import_paths() -> None:
    hf_home = REPO_ROOT / "eval" / ".hf_cache"
    os.environ["HF_HOME"] = str(hf_home)
    for entry in (str(CATS_CONVERTER), str(REPO_ROOT)):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _load_normalized_tools_by_hash(converted: list[dict]) -> dict[str, dict]:
    """Re-derive normalized BFCL tools (records.jsonl stores hashes only, not schemas)."""
    _ensure_import_paths()
    import cats  # noqa: WPS433
    from eval.part1.pipeline import (  # noqa: WPS433
        convert_unique_tools,
        load_bfcl_tools,
        validate_and_aggregate,
    )
    from from_json import normalize_map_python_types  # noqa: WPS433

    pool_hashes = {r["tool_hash"] for r in converted}
    print(
        "Loading normalized tool schemas from BFCL via eval.part1.pipeline "
        "(records.jsonl has no stored schema; same map→validate→dedupe path as Part 1)…"
    )
    occurrences = load_bfcl_tools()
    aggregates, invalid_entries, _ = validate_and_aggregate(
        occurrences,
        normalize_map_python_types=normalize_map_python_types,
    )
    invalid_hashes = {agg.tool_hash for agg, _ in invalid_entries}
    invalid_reasons = {agg.tool_hash: reason for agg, reason in invalid_entries}
    results = convert_unique_tools(
        aggregates,
        convert_with_report=cats.convert_with_report,
        invalid_hashes=invalid_hashes,
        invalid_reasons=invalid_reasons,
    )
    index = {
        r.tool_hash: r.normalized_tool
        for r in results
        if r.normalized_tool is not None and r.tool_hash in pool_hashes
    }
    missing = pool_hashes - set(index)
    if missing:
        raise SystemExit(
            f"normalized schemas missing for {len(missing)} converted record(s); re-run Part 1."
        )
    print(f"  matched {len(index)} converted tool_hashes\n")
    return index


def _pretty_reductions_for(
    converted: list[dict],
    tok: str,
    normalized_by_hash: dict[str, dict],
    count_fn: Callable[[str], int],
) -> list[float]:
    from eval.part1.pipeline import pretty_tool_json  # noqa: WPS433

    ckey = f"tokens_cats_{tok}"
    reductions: list[float] = []
    for row in converted:
        cats_tokens = row.get(ckey)
        if cats_tokens is None:
            continue
        normalized = normalized_by_hash.get(row["tool_hash"])
        if normalized is None:
            continue
        pretty_tokens = count_fn(pretty_tool_json(normalized))
        if pretty_tokens <= 0:
            continue
        reductions.append((pretty_tokens - cats_tokens) / pretty_tokens * 100)
    return sorted(reductions)


def _pretty_reductions_for_anthropic(
    converted: list[dict],
    pretty_json_counts: dict[str, int],
) -> list[float]:
    reductions: list[float] = []
    for row in converted:
        cats_tokens = row.get("tokens_cats_anthropic")
        if cats_tokens is None:
            continue
        pretty_tokens = pretty_json_counts.get(row["tool_hash"])
        if pretty_tokens is None or pretty_tokens <= 0:
            continue
        reductions.append((pretty_tokens - cats_tokens) / pretty_tokens * 100)
    return sorted(reductions)


def _fetch_pretty_anthropic_counts(
    converted: list[dict],
    normalized_by_hash: dict[str, dict],
    *,
    max_workers: int,
) -> dict[str, int]:
    from eval.part1.anthropic_cache import AnthropicPrettyTokenCache  # noqa: WPS433
    from eval.part1.anthropic_fetch import fetch_anthropic_pretty_json_counts  # noqa: WPS433
    from eval.part1.pipeline import pretty_tool_json  # noqa: WPS433

    if not os.environ.get("ANTHROPIC_API_KEY_PART1"):
        raise SystemExit(
            "ANTHROPIC_API_KEY_PART1 is required for --with-anthropic-pretty "
            "(set in environment or .env at repo root)."
        )

    missing_cats = [
        row["tool_hash"]
        for row in converted
        if row.get("tokens_cats_anthropic") is None
    ]
    if missing_cats:
        raise SystemExit(
            f"{len(missing_cats)} converted tool(s) lack tokens_cats_anthropic in records.jsonl; "
            "re-run Part 1 with --with-anthropic first."
        )

    cache_path = REPO_ROOT / "eval" / "cache" / "anthropic_pretty_tokens.json"
    cache = AnthropicPrettyTokenCache(cache_path)
    entries: list[tuple[str, str, int]] = []
    for row in converted:
        normalized = normalized_by_hash.get(row["tool_hash"])
        if normalized is None:
            continue
        entries.append(
            (
                row["tool_hash"],
                pretty_tool_json(normalized),
                int(row["tokens_cats_anthropic"]),
            )
        )

    print(f"Fetching Anthropic pretty-printed JSON counts (cached at {cache_path})…")
    fetch_anthropic_pretty_json_counts(entries, cache, max_workers=max_workers)
    cache.save()

    counts: dict[str, int] = {}
    for tool_hash, _, _ in entries:
        cached = cache.get(tool_hash)
        if cached is not None:
            counts[tool_hash] = cached
    return counts


def _load_pretty_anthropic_counts_from_cache(converted: list[dict]) -> dict[str, int]:
    cache_path = REPO_ROOT / "eval" / "cache" / "anthropic_pretty_tokens.json"
    if not cache_path.is_file():
        return {}
    from eval.part1.anthropic_cache import AnthropicPrettyTokenCache  # noqa: WPS433

    cache = AnthropicPrettyTokenCache(cache_path)
    return {
        row["tool_hash"]: count
        for row in converted
        if (count := cache.get(row["tool_hash"])) is not None
    }


def _anthropic_pretty_complete(converted: list[dict], pretty_json_counts: dict[str, int]) -> bool:
    return bool(pretty_json_counts) and all(
        row.get("tokens_cats_anthropic") is not None
        and pretty_json_counts.get(row["tool_hash"]) is not None
        for row in converted
    )


def _combined_reductions(by_tok: dict[str, list[float]], tokenizers: tuple[str, ...]) -> list[float]:
    return sorted(v for tok in tokenizers for v in by_tok.get(tok, []))


def _write_compact_vs_pretty_combined_boxplot(
    compact_combined: list[float],
    pretty_combined: list[float],
    out_img: Path,
) -> None:
    compact_med = _percentile(compact_combined, 50)
    pretty_med = _percentile(pretty_combined, 50)
    compact_mean = statistics.mean(compact_combined)
    pretty_mean = statistics.mean(pretty_combined)
    print("== compact vs pretty (all 3 tokenizers combined) ==")
    print(f"  compact JSON     : median {compact_med:6.2f}%  mean {compact_mean:6.2f}%  (n={len(compact_combined)})")
    print(f"  pretty-printed   : median {pretty_med:6.2f}%  mean {pretty_mean:6.2f}%  (n={len(pretty_combined)})")
    print(
        f"  gap (pretty - compact): +{pretty_med - compact_med:.1f}pp median, "
        f"+{pretty_mean - compact_mean:.1f}pp mean\n"
    )

    _write_boxplot(
        [compact_combined, pretty_combined],
        [
            f"compact JSON\n(all 3 tokenizers)\n(n={len(compact_combined)})",
            f"pretty-printed JSON\n(all 3 tokenizers)\n(n={len(pretty_combined)})",
        ],
        out_img,
        kind="percent",
        title="CATS token reduction: compact vs pretty-printed JSON (all tokenizers pooled)",
        ylabel="Per-tool token reduction (%)  --  higher = CATS smaller",
    )


def _run_pretty_printed_analysis(
    converted: list[dict],
    compact_by_tok: dict[str, list[float]],
    out_pretty: Path,
    out_compact_vs_pretty: Path,
    *,
    with_anthropic_pretty: bool,
    anthropic_workers: int,
) -> None:
    _ensure_import_paths()
    from eval.part1.tokenizers import (  # noqa: WPS433
        default_qwen_counter,
        default_tiktoken_counter,
    )

    counters: dict[str, Callable[[str], int]] = {
        "tiktoken": default_tiktoken_counter().count,
        "qwen": default_qwen_counter().count,
    }

    print("=" * 72)
    print("Pretty-printed JSON baseline (robustness; compact JSON remains primary)")
    print("  Pretty bytes: json.dumps(normalized_tool, indent=2, sort_keys=True)")
    print("  CATS token counts: unchanged from records.jsonl")
    if with_anthropic_pretty:
        print("  anthropic: fetching pretty-printed JSON via API (--with-anthropic-pretty)\n")
    else:
        print(
            "  anthropic: use --with-anthropic-pretty to fetch, or rely on "
            "eval/cache/anthropic_pretty_tokens.json if already populated\n"
        )

    normalized_by_hash = _load_normalized_tools_by_hash(converted)

    pretty_anthropic_counts: dict[str, int] = {}
    if with_anthropic_pretty:
        pretty_anthropic_counts = _fetch_pretty_anthropic_counts(
            converted,
            normalized_by_hash,
            max_workers=anthropic_workers,
        )
    else:
        pretty_anthropic_counts = _load_pretty_anthropic_counts_from_cache(converted)
        if _anthropic_pretty_complete(converted, pretty_anthropic_counts):
            print(
                f"  anthropic pretty-print: loaded {len(pretty_anthropic_counts)} counts "
                "from eval/cache/anthropic_pretty_tokens.json\n"
            )

    pretty_tokenizers = list(PRETTY_LOCAL_TOKENIZERS)
    if _anthropic_pretty_complete(converted, pretty_anthropic_counts):
        pretty_tokenizers.append("anthropic")

    pretty_box_data: list[list[float]] = []
    pretty_box_labels: list[str] = []
    all_pretty: list[float] = []
    headline_parts: list[str] = []
    pretty_by_tok: dict[str, list[float]] = {}

    for tok in pretty_tokenizers:
        compact = compact_by_tok.get(tok)
        if not compact:
            print(f"== {tok} ==\n  skipped (no compact baseline in records)\n")
            continue

        if tok == "anthropic":
            pretty = _pretty_reductions_for_anthropic(converted, pretty_anthropic_counts)
        else:
            pretty = _pretty_reductions_for(converted, tok, normalized_by_hash, counters[tok])
        if len(pretty) != len(compact):
            print(
                f"WARNING: {tok} pretty n={len(pretty)} != compact n={len(compact)}",
                file=sys.stderr,
            )

        compact_med = _percentile(compact, 50)
        compact_mean = statistics.mean(compact)
        pretty_med = _percentile(pretty, 50)
        pretty_mean = statistics.mean(pretty)
        med_gap = pretty_med - compact_med
        mean_gap = pretty_mean - compact_mean

        print(f"== {tok} ({BOX_LABELS[tok]}) ==  (n = {len(pretty)})")
        print("  Per-tool % reduction vs pretty-printed normalized JSON:")
        print(f"    median : {pretty_med:6.2f}%")
        print(f"    mean   : {pretty_mean:6.2f}%")
        print(
            f"  vs compact baseline: {compact_med:.1f}% median / {compact_mean:.1f}% mean"
            f"  ->  {pretty_med:.1f}% / {pretty_mean:.1f}%"
            f"  (+{med_gap:.1f}pp median, +{mean_gap:.1f}pp mean)\n"
        )

        headline_parts.append(f"{tok} +{med_gap:.1f}pp median")
        pretty_by_tok[tok] = pretty
        pretty_box_data.append(pretty)
        pretty_box_labels.append(f"{BOX_LABELS[tok]}\n(n={len(pretty)})")
        all_pretty.extend(pretty)

    if all_pretty:
        combined = sorted(all_pretty)
        compact_combined = sorted(
            v for tok in pretty_tokenizers for v in compact_by_tok.get(tok, [])
        )
        if compact_combined:
            label = "all pretty tokenizers" if len(pretty_tokenizers) > 2 else "tiktoken+qwen combined"
            print(f"== {label} ==  (n = {len(combined)})")
            print(f"    median : {_percentile(combined, 50):6.2f}%")
            print(f"    mean   : {statistics.mean(combined):6.2f}%")
            med_gap = _percentile(combined, 50) - _percentile(compact_combined, 50)
            mean_gap = statistics.mean(combined) - statistics.mean(compact_combined)
            print(
                f"  vs compact combined: +{med_gap:.1f}pp median, +{mean_gap:.1f}pp mean\n"
            )
        pretty_box_data.append(combined)
        pretty_box_labels.append(f"all tokenizers combined\n(n={len(combined)})")

    if headline_parts:
        skipped = ""
        if "anthropic" not in pretty_tokenizers:
            skipped = " (anthropic pretty-print unavailable)"
        print(
            "METHODS (pretty-printed robustness): "
            + "; ".join(headline_parts)
            + f" vs compact baseline{skipped}."
        )
        print()

    if not pretty_box_data:
        print("No pretty-printed box plot written.")
        return

    _write_boxplot(
        pretty_box_data,
        pretty_box_labels,
        out_pretty,
        kind="percent",
        title="CATS per-tool token reduction vs pretty-printed JSON",
        ylabel="Per-tool token reduction (%) vs pretty JSON  --  higher = CATS smaller",
    )

    if all(tok in compact_by_tok and tok in pretty_by_tok for tok in TOKENIZERS):
        _write_compact_vs_pretty_combined_boxplot(
            _combined_reductions(compact_by_tok, TOKENIZERS),
            _combined_reductions(pretty_by_tok, TOKENIZERS),
            out_compact_vs_pretty,
        )
    else:
        missing = [tok for tok in TOKENIZERS if tok not in compact_by_tok or tok not in pretty_by_tok]
        print(
            f"Skipping compact-vs-pretty combined box plot "
            f"(missing data for: {', '.join(missing)}).\n"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Token-reduction statistics and box plots for Part 1 records.jsonl",
    )
    parser.add_argument(
        "records",
        nargs="?",
        type=Path,
        default=Path("eval/out/records.jsonl"),
        help="Path to records.jsonl (default: eval/out/records.jsonl)",
    )
    parser.add_argument(
        "out_percent",
        nargs="?",
        type=Path,
        default=None,
        help="Compact percent box plot path (default: beside records.jsonl)",
    )
    parser.add_argument(
        "--with-anthropic-pretty",
        action="store_true",
        help="Fetch Anthropic counts for pretty-printed JSON (cached; requires API key)",
    )
    parser.add_argument(
        "--anthropic-workers",
        type=int,
        default=2,
        help="Concurrent Anthropic count_tokens workers (global rate limit is the bottleneck)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = args.records
    out_percent = args.out_percent or path.parent / "reduction_boxplot.png"
    out_absolute = path.parent / "absolute_reduction_boxplot.png"
    out_pretty = path.parent / "pretty_printed_reduction_boxplot.png"
    out_compact_vs_pretty = path.parent / "compact_vs_pretty_combined_boxplot.png"

    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    converted = [r for r in rows if r["bucket"] == "converted"]
    print(f"records: {len(rows)}  |  converted: {len(converted)}\n")

    percent_box_data: list[list[float]] = []
    percent_box_labels: list[str] = []
    absolute_box_data: list[list[float]] = []
    absolute_box_labels: list[str] = []
    all_reductions: list[float] = []
    all_deltas: list[float] = []
    compact_by_tok: dict[str, list[float]] = {}

    for tok in TOKENIZERS:
        reductions = _reductions_for(converted, tok)
        if not reductions:
            print(f"== {tok} ==")
            print("  no data (column not populated -- did you run --with-anthropic?)\n")
            continue

        deltas = _absolute_deltas_for(converted, tok)
        agg = _aggregate_for(converted, tok)
        _print_tokenizer_stats(tok, reductions, deltas)
        if agg:
            print(
                f"  Corpus-wide aggregate reduction: {agg[0]:6.2f}%"
                f"  (sum json {agg[1]} -> sum cats {agg[2]})\n"
            )

        compact_by_tok[tok] = reductions
        percent_box_data.append(reductions)
        percent_box_labels.append(f"{BOX_LABELS[tok]}\n(n={len(reductions)})")
        absolute_box_data.append(deltas)
        absolute_box_labels.append(f"{BOX_LABELS[tok]}\n(n={len(deltas)})")
        all_reductions.extend(reductions)
        all_deltas.extend(deltas)

    if all_reductions:
        all_reductions.sort()
        combined_deltas = sorted(all_deltas)
        print(f"== all (combined) ==  (n = {len(all_reductions)})")
        _print_reduction_stats("all (combined)", all_reductions, show_heading=False)
        _print_absolute_stats("all (combined)", combined_deltas, show_heading=False)
        percent_box_data.append(all_reductions)
        percent_box_labels.append(f"all tokenizers combined\n(n={len(all_reductions)})")
        absolute_box_data.append(combined_deltas)
        absolute_box_labels.append(f"all tokenizers combined\n(n={len(combined_deltas)})")

    if not percent_box_data:
        print("No populated tokenizer columns -- no box plots written.")
        return

    _write_boxplot(percent_box_data, percent_box_labels, out_percent, kind="percent")
    _write_boxplot(absolute_box_data, absolute_box_labels, out_absolute, kind="absolute")
    _run_pretty_printed_analysis(
        converted,
        compact_by_tok,
        out_pretty,
        out_compact_vs_pretty,
        with_anthropic_pretty=args.with_anthropic_pretty,
        anthropic_workers=args.anthropic_workers,
    )


if __name__ == "__main__":
    main()
