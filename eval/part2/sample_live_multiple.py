#!/usr/bin/env python3
"""Reproduce the Part 2 live_multiple subsample draw and verify against the commit.

The committed IDs in live_multiple_subsample.json were drawn as:

    rng = numpy.random.default_rng(seed)
    indices = rng.choice(N, size=n, replace=False)
    selected = [ids[i] for i in indices]

where ``ids`` is the live_multiple entry-ID list in BFCL prompt-file order
(bfcl-eval 2026.3.23; N = 1053), and ``seed`` / ``n`` are read from the
committed JSON. Equivalent forms (choice over an id array, or choice over
``np.arange(N)``) produce the same set; this is the index form.

READ-ONLY with respect to live_multiple_subsample.json — never writes it.

The draw was verified under numpy 1.26.4, the version recorded in the committed
JSON. NumPy guarantees the PCG64 bit stream across versions but does not freeze
``Generator`` method implementations the way legacy ``RandomState`` is frozen, so
a future major release could in principle change ``choice``. The assertion below
is what protects against that: a mismatch means the numpy version, not the data.

Run from the repo root:

    uv run --project cats-converter python eval/part2/sample_live_multiple.py

Exit 0 if the draw matches the committed selected_ids (as sorted lists);
exit nonzero on mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cats-converter"))

import numpy as np  # noqa: E402

from eval.part2.corpus import SUBSAMPLE_PATH, _read_jsonl  # noqa: E402
from eval.part2.paths import bfcl_prompt_path  # noqa: E402

CATEGORY = "live_multiple"


def main() -> None:
    payload = json.loads(SUBSAMPLE_PATH.read_text(encoding="utf-8"))
    seed = int(payload["seed"])
    n = int(payload["n"])
    expected = sorted(payload["selected_ids"])
    source_total = int(payload["source_total"])

    ids = [row["id"] for row in _read_jsonl(bfcl_prompt_path(CATEGORY))]
    if len(ids) != source_total:
        raise SystemExit(
            f"population size {len(ids)} != committed source_total {source_total}"
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ids), size=n, replace=False)
    drawn = sorted(ids[i] for i in indices)

    if drawn != expected:
        overlap = len(set(drawn) & set(expected))
        raise SystemExit(
            f"draw mismatch: {overlap}/{n} overlap with committed selected_ids"
        )

    print(
        f"OK: seed={seed} n={n} source_total={source_total} "
        f"bfcl_file_order + rng.choice(N, size=n, replace=False) "
        f"matches committed selected_ids"
    )


if __name__ == "__main__":
    main()
