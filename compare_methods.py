"""How the planner's selector compares with standard ways of picking EIS
frequencies, under the SAME protocol as benchmark.py.

Same 7 cells, same leave-one-cell-out folds, same commissioning-referenced
features, same final GP model, same time budgets, at most 3 frequencies.
Only the rule that picks the frequencies differs, and every rule sees only
the 6 training cells of its fold:

  greedy_cv    this repo: forward selection scored by inner leave-one-cell-out
  corr_rank    rank frequencies by |Pearson r| with SOH, take the top ones that fit
  lasso_path   L1 path on the 2 features per affordable frequency; first
               frequencies to enter the path
  ard_rank     GP with one lengthscale per feature (ARD), as in Zhang et al.;
               frequencies with the shortest lengthscales first

Also: the original paper's own pipeline (raw spectra, all 60 frequencies,
isotropic GP, trained on the six other cells, tested on 35C02), scored the
way its Fig. 3a plots it (estimate / first estimate vs measured / first).

  python compare_methods.py      (~10 min on 2 cores)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler

from twcx.data import ZHANG_FREQS, load_zhang2020
from twcx.model import FadeModel, features, mae_pp, prepare, seconds_per_freq

BUDGETS = [0.5, 2.0]
MAX_K = 3


def _train_xy(train, idx, stride=4):
    X = np.vstack([features(p, idx)[p.eval_mask][::stride] for p in train])
    y = np.hstack([p.soh[p.eval_mask][::stride] for p in train])
    return X, y


def _take(order, cost, budget):
    chosen, spent = [], 0.0
    for j in order:
        if len(chosen) == MAX_K:
            break
        if spent + cost[j] <= budget:
            chosen.append(int(j))
            spent += cost[j]
    return chosen


def corr_rank(train, cost, budget):
    afford = [j for j in range(len(cost)) if cost[j] <= budget]
    X, y = _train_xy(train, afford)
    n = len(afford)
    score = []
    for k in range(n):
        r1 = abs(np.corrcoef(X[:, k], y)[0, 1])
        r2 = abs(np.corrcoef(X[:, n + k], y)[0, 1])
        score.append(np.nan_to_num(max(r1, r2)))
    order = [afford[k] for k in np.argsort(score)[::-1]]
    return _take(order, cost, budget)


def lasso_rank(train, cost, budget):
    afford = [j for j in range(len(cost)) if cost[j] <= budget]
    X, y = _train_xy(train, afford)
    X = StandardScaler().fit_transform(X)
    _, coefs, _ = lasso_path(X, y - y.mean(), n_alphas=200, eps=1e-4)
    n = len(afford)
    order = []
    for a in range(coefs.shape[1]):              # alphas from large to small
        for k in np.flatnonzero(coefs[:, a]):
            j = afford[k % n]
            if j not in order:
                order.append(j)
    return _take(order, cost, budget)


def ard_rank(train, cost, budget):
    afford = [j for j in range(len(cost)) if cost[j] <= budget]
    X, y = _train_xy(train, afford, stride=6)
    X = StandardScaler().fit_transform(X)
    k = ConstantKernel() * RBF(np.ones(X.shape[1]), (1e-2, 1e4)) + WhiteKernel(0.05, (1e-5, 1))
    gp = GaussianProcessRegressor(k, normalize_y=True, random_state=0).fit(X, y)
    ls = gp.kernel_.k1.k2.length_scale
    n = len(afford)
    per_freq = np.minimum(ls[:n], ls[n:])
    order = [afford[i] for i in np.argsort(per_freq)]
    return _take(order, cost, budget)


RULES = {"corr_rank": corr_rank, "lasso_path": lasso_rank, "ard_rank": ard_rank}


def run_fold(args):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    i, preps, cost = args
    test, train = preps[i], preps[:i] + preps[i + 1:]
    out = {"cell": test.name}
    for B in BUDGETS:
        for name, rule in RULES.items():
            idx = rule(train, cost, B)
            m = FadeModel(idx, stride=4).fit(train)
            out[f"{name}@{B:g}s"] = {"mae_pp": mae_pp(m, test),
                                     "freqs_hz": [round(float(ZHANG_FREQS[j]), 4) for j in idx],
                                     "seconds": float(cost[idx].sum()) if idx else 0.0}
    print("fold", test.name, "done", flush=True)
    return out


def zhang_pipeline(cells):
    """The authors' Multi_T_EIS_Capacity_GPR.m, in Python: z-scored raw
    spectra of the six training cells, isotropic SE GP, predict 35C02."""
    train = [c for c in cells if c.name != "35C02"]
    test = [c for c in cells if c.name == "35C02"][0]
    X = np.vstack([np.hstack([c.re, c.neg_im]) for c in train])
    y = np.hstack([c.capacity for c in train])
    sc = StandardScaler().fit(X)
    gp = GaussianProcessRegressor(ConstantKernel() * RBF() + WhiteKernel(0.01), random_state=0)
    gp.fit(sc.transform(X), y)
    pred = gp.predict(sc.transform(np.hstack([test.re, test.neg_im])))
    rel_pred, rel_true = pred / pred[0], test.capacity / test.capacity[0]
    return {"abs_mae_mAh": float(np.mean(np.abs(pred - test.capacity))),
            "fig3a_relative_mae_pp": float(np.mean(np.abs(rel_pred - rel_true)) * 100),
            "note": "Fig. 3a normalises both curves by their first point; this is that error"}


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    cells = load_zhang2020()
    preps = [prepare(c) for c in cells]
    cost = seconds_per_freq(ZHANG_FREQS)
    with mp.Pool(min(len(preps), os.cpu_count() or 1)) as pool:
        folds = pool.map(run_fold, [(i, preps, cost) for i in range(len(preps))])

    bench = json.load(open("results/benchmark.json"))
    bf = {f["cell"]: f for f in bench["folds"]}
    table = {}
    for B in BUDGETS:
        key = f"twcx@{B:g}s"
        per = {c: bf[c]["arms"][key]["mae_pp"] for c in bf}
        table[f"greedy_cv@{B:g}s"] = {"mean_mae_pp": float(np.mean(list(per.values()))), "per_cell": per,
                                      "mean_seconds": float(np.mean([bf[c]["arms"][key]["seconds"] for c in bf]))}
        for name in RULES:
            k = f"{name}@{B:g}s"
            per = {f["cell"]: f[k]["mae_pp"] for f in folds}
            table[k] = {"mean_mae_pp": float(np.mean(list(per.values()))), "per_cell": per,
                        "mean_seconds": float(np.mean([f[k]["seconds"] for f in folds])),
                        "greedy_wins_cells": sum(table[f"greedy_cv@{B:g}s"]["per_cell"][c] < per[c] for c in per)}
    res = {"table": table, "folds": folds, "zhang_pipeline_on_35C02": zhang_pipeline(cells),
           "twcx_on_35C02": {B: bf["35C02"]["arms"][f"twcx@{B:g}s"]["mae_pp"] for B in BUDGETS},
           "full_sweep_on_35C02": bf["35C02"]["arms"]["full_sweep"]["mae_pp"]}
    with open("results/compare_methods.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n{'rule':<18}{'MAE pp':>8}{'s/test':>8}{'greedy wins':>13}")
    for k, v in table.items():
        print(f"{k:<18}{v['mean_mae_pp']:8.2f}{v['mean_seconds']:8.2f}{str(v.get('greedy_wins_cells', '')):>13}")
    print(json.dumps(res["zhang_pipeline_on_35C02"], indent=1), res["twcx_on_35C02"], res["full_sweep_on_35C02"])


if __name__ == "__main__":
    main()
