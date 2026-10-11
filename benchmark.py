"""Held-out benchmark on real laboratory data (Zhang et al. 2020).

Protocol, fixed before the run:
  * 7 cells. Each cell is held out once (leave-one-cell-out, 7 folds).
  * Frequency choice for a fold uses ONLY the other 6 cells (inner
    leave-one-cell-out). The held-out cell's spectra never touch selection.
  * Target: SOH = capacity / commissioning capacity (mean of first 5 checks).
  * Metric: mean absolute SOH error, percentage points, post-commissioning.
  * Decision metric: flag SOH < 80% (the usual end-of-first-life line).
  * Cost: estimated seconds per test, (1 settle + 2 measured periods)/f + 0.2 s.

Arms:
  nothing        predict the training mean (no measurement, 0 s)
  ac_ir_953Hz    the grid point nearest the 1 kHz AC-IR standard
  full_sweep     all 60 frequencies, 20 kHz -> 0.02 Hz
  twcx@B         greedy selection under per-test budget B seconds
  random@B       expected error of one random affordable frequency
  hindsight@B    best single affordable frequency chosen WITH the answer
                 (not achievable; shows how much selection leaves on the table)

Run:  python benchmark.py            (about 10-15 min on 2 cores)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from twcx.data import ZHANG_FREQS, load_zhang2020
from twcx.model import FadeModel, mae_pp, prepare, seconds_per_freq
from twcx.select import CVScorer, greedy_budget

BUDGETS = [0.25, 0.5, 2.0, 10.0, 60.0]
IDX_1K = int(np.argmin(np.abs(ZHANG_FREQS - 1000.0)))
THRESH = 0.80


def run_fold(args):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    i, preps, cost = args
    test = preps[i]
    train = preps[:i] + preps[i + 1:]
    out = {"cell": test.name, "arms": {}, "selected": {}, "singles_mae_pp": []}
    t0 = time.time()

    def final(idx):
        m = FadeModel(idx, stride=4).fit(train)
        pred = m.predict(test)
        return {"mae_pp": mae_pp(m, test),
                "seconds": float(cost[list(idx)].sum()) if idx else 0.0,
                "freqs_hz": [round(float(ZHANG_FREQS[j]), 4) for j in idx],
                "pred": pred[test.eval_mask].round(5).tolist()}

    out["arms"]["nothing"] = final([])
    out["arms"]["ac_ir_953Hz"] = final([IDX_1K])
    out["arms"]["full_sweep"] = final(list(range(len(cost))))

    singles = [final([j])["mae_pp"] for j in range(len(cost))]
    out["singles_mae_pp"] = singles

    scorer = CVScorer(train, stride=8)
    for B in BUDGETS:
        idx, hist = greedy_budget(scorer, cost, B)
        out["arms"][f"twcx@{B:g}s"] = final(idx)
        out["selected"][f"{B:g}"] = hist
        afford = [j for j in range(len(cost)) if cost[j] <= B]
        out["arms"][f"random@{B:g}s"] = {
            "mae_pp": float(np.mean([singles[j] for j in afford])),
            "seconds": float(np.mean(cost[afford]))}
        jb = min(afford, key=lambda j: singles[j])
        out["arms"][f"hindsight@{B:g}s"] = {"mae_pp": singles[jb], "seconds": float(cost[jb]),
                                           "freqs_hz": [round(float(ZHANG_FREQS[jb]), 4)]}
    out["truth"] = test.soh[test.eval_mask].round(5).tolist()
    out["runtime_s"] = round(time.time() - t0, 1)
    print(f"fold {test.name}: done in {out['runtime_s']} s", flush=True)
    return out


def absolute_capacity_check(cells):
    """Side check behind the task choice: predicting ABSOLUTE capacity of an
    unseen cell from its raw spectrum (no reference)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.preprocessing import StandardScaler

    def F(c, idx):
        return np.hstack([c.re[:, idx], c.neg_im[:, idx]])

    res = {}
    for lab, idx in [("nothing", None), ("953Hz", [IDX_1K]), ("17.8Hz", [30]), ("full_sweep", list(range(60)))]:
        errs = {}
        for i, c in enumerate(cells):
            tr = cells[:i] + cells[i + 1:]
            ytr = np.hstack([t.capacity[::4] for t in tr])
            if idx is None:
                errs[c.name] = float(np.mean(np.abs(ytr.mean() - c.capacity)))
                continue
            Xtr = np.vstack([F(t, idx)[::4] for t in tr])
            sc = StandardScaler().fit(Xtr)
            gp = GaussianProcessRegressor(ConstantKernel() * RBF() + WhiteKernel(0.05), normalize_y=True)
            gp.fit(sc.transform(Xtr), ytr)
            errs[c.name] = float(np.mean(np.abs(gp.predict(sc.transform(F(c, idx))) - c.capacity)))
        res[lab] = {"per_cell_mae_mAh": errs, "mean_mae_mAh": float(np.mean(list(errs.values())))}
    return res


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    cells = load_zhang2020()
    preps = [prepare(c) for c in cells]
    cost = seconds_per_freq(ZHANG_FREQS)
    t0 = time.time()
    with mp.Pool(min(len(preps), os.cpu_count() or 1)) as pool:
        folds = pool.map(run_fold, [(i, preps, cost) for i in range(len(preps))])

    arms = list(folds[0]["arms"])
    summary = {}
    for a in arms:
        per = {f["cell"]: f["arms"][a]["mae_pp"] for f in folds}
        row = {"mean_mae_pp": float(np.mean(list(per.values()))), "per_cell_mae_pp": per,
               "mean_seconds_per_test": float(np.mean([f["arms"][a]["seconds"] for f in folds]))}
        if "pred" in folds[0]["arms"][a]:
            tp = fp = tn = fn = 0
            for f in folds:
                y = np.array(f["truth"]) < THRESH
                p = np.array(f["arms"][a]["pred"]) < THRESH
                tp += int((y & p).sum()); fp += int((~y & p).sum())
                tn += int((~y & ~p).sum()); fn += int((y & ~p).sum())
            sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
            row["eol_flag"] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                               "sensitivity": sens, "specificity": spec,
                               "balanced_accuracy": (sens + spec) / 2}
        summary[a] = row

    # paired cell-level comparison against the two practical alternatives
    def wins(a, b):
        return sum(summary[a]["per_cell_mae_pp"][c] < summary[b]["per_cell_mae_pp"][c] for c in summary[a]["per_cell_mae_pp"])

    paired = {}
    for B in BUDGETS:
        a = f"twcx@{B:g}s"
        paired[a] = {"beats_nothing_cells": wins(a, "nothing"),
                     "beats_ac_ir_cells": wins(a, "ac_ir_953Hz"),
                     "beats_full_sweep_cells": wins(a, "full_sweep"),
                     "beats_random_cells": wins(a, f"random@{B:g}s"),
                     "n_cells": len(folds)}

    result = {
        "dataset": "Zhang et al. 2020 Nat Commun 11:1706 — Eunicell LR2032 coin cells, EIS at state V",
        "cells": [{"name": c.name, **c.meta, "n_tests": c.n,
                   "soh_range": [round(float(p.soh.min()), 3), round(float(p.soh.max()), 3)]}
                  for c, p in zip(cells, preps)],
        "frequency_grid_hz": [round(float(f), 4) for f in ZHANG_FREQS],
        "cost_seconds": cost.round(3).tolist(),
        "full_sweep_seconds": float(cost.sum()),
        "budgets_s": BUDGETS,
        "summary": summary,
        "paired": paired,
        "folds": folds,
        "absolute_capacity_check": absolute_capacity_check(cells),
        "runtime_s": round(time.time() - t0, 1),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark.json", "w") as f:
        json.dump(result, f, indent=1)

    print(f"\n{'arm':<18}{'MAE pp':>8}{'s/test':>9}{'bal.acc':>9}")
    for a in arms:
        s = summary[a]
        ba = s.get("eol_flag", {}).get("balanced_accuracy")
        print(f"{a:<18}{s['mean_mae_pp']:8.2f}{s['mean_seconds_per_test']:9.2f}"
              f"{'' if ba is None else f'{ba:9.3f}'}")
    print(json.dumps(paired, indent=1))


if __name__ == "__main__":
    main()
