"""twcx — pick the impedance frequencies worth their test time.

  python -m twcx plan my_eis.csv --budget 2       # your data
  python -m twcx plan --zhang --budget 2          # the public demo data
  python -m twcx frontier my_eis.csv              # error vs time across budgets

Selection is scored by leave-one-cell-out error on the cells you supply.
The more cells, the more the number means: with fewer than ~5 cells treat
the output as a shortlist to verify, not a qualified test plan.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from .data import load_long_csv, load_zhang2020
from .model import prepare, seconds_per_freq
from .select import CVScorer, greedy_budget


def _load(a):
    if a.zhang:
        return load_zhang2020()
    if not a.csv:
        sys.exit("give a CSV path or --zhang")
    return load_long_csv(a.csv)


def _ref_points(scorer, cost, freqs):
    i1k = int(np.argmin(np.abs(freqs - 1000.0)))
    return {
        "nothing": {"cv_mae_pp": scorer(()), "seconds": 0.0},
        f"single_{freqs[i1k]:.4g}Hz_(nearest_1kHz)": {"cv_mae_pp": scorer((i1k,)), "seconds": float(cost[i1k])},
        "full_sweep": {"cv_mae_pp": scorer(tuple(range(len(freqs)))), "seconds": float(cost.sum())},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="twcx", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "frontier"])
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--zhang", action="store_true", help="use the public Zhang 2020 cells")
    ap.add_argument("--budget", type=float, default=2.0, help="seconds per test (plan)")
    ap.add_argument("--k0", type=int, default=5, help="commissioning tests per cell")
    ap.add_argument("--settle", type=float, default=1.0, help="settling periods per frequency")
    ap.add_argument("--periods", type=float, default=2.0, help="measured periods per frequency")
    ap.add_argument("--overhead", type=float, default=0.2, help="fixed seconds per frequency")
    ap.add_argument("--stride", type=int, default=8, help="use every n-th test when scoring")
    ap.add_argument("--out", help="write the plan / frontier JSON here")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    cells = _load(a)
    if len(cells) < 3:
        sys.exit("need at least 3 cells: selection is scored on cells it was not fitted to")
    preps = [prepare(c, a.k0) for c in cells]
    freqs = cells[0].freqs
    cost = seconds_per_freq(freqs, a.settle, a.periods, a.overhead)
    scorer = CVScorer(preps, stride=a.stride)
    refs = _ref_points(scorer, cost, freqs)

    budgets = [a.budget] if a.cmd == "plan" else [0.25, 0.5, 2, 10, 60, float(cost.sum())]
    rows = []
    for B in budgets:
        idx, hist = greedy_budget(scorer, cost, B)
        rows.append({"budget_s": B, "freqs_hz": [float(freqs[j]) for j in idx],
                     "seconds_per_test": float(cost[idx].sum()) if idx else 0.0,
                     "cv_mae_pp": hist[-1]["cv_mae_pp"], "steps": hist})

    report = {"n_cells": len(cells), "cells": [c.name for c in cells],
              "cost_model": {"settle_periods": a.settle, "measured_periods": a.periods,
                             "overhead_s": a.overhead},
              "reference_points": refs, "plans": rows,
              "caveat": ("cv_mae_pp is leave-one-cell-out SOH error in percentage points on the "
                         "cells supplied. For the CHOSEN plan it is optimistic, because the same "
                         "cells also chose the frequencies; benchmark.py's nested protocol gives "
                         "the honest held-out figure")}

    print(f"{len(cells)} cells, {len(freqs)} frequencies, full sweep ≈ {cost.sum():.0f} s/test\n")
    for k, v in refs.items():
        print(f"  {k:<34} CV error {v['cv_mae_pp']:6.2f} pp   {v['seconds']:8.2f} s")
    print()
    for r in rows:
        fs = ", ".join(f"{f:.4g} Hz" for f in r["freqs_hz"]) or "(measure nothing)"
        print(f"  budget {r['budget_s']:>7.2f} s -> {fs:<34} CV error {r['cv_mae_pp']:6.2f} pp   "
              f"{r['seconds_per_test']:6.2f} s")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"\nwrote {a.out}")
    return report


if __name__ == "__main__":
    main()
