"""Extra baselines Sol asked for, same protocol as benchmark.py:

  fixed_17.8Hz       the single frequency Zhang et al. flag (grid index 30)
  fixed_17.8+2.16Hz  the published pair (indices 30 and 39), 1.96 s
  full_sweep_ard     all 60 frequencies with an ARD kernel (one lengthscale
                     per feature), the fair version of the full-sweep model;
                     benchmark.py's full sweep uses one shared lengthscale

No selection happens here, so there is no leakage question: the fixed
choices come from the literature, the ARD model sees only training cells.
Merges into results/compare_methods.json under "extra".

  python compare_extra.py
"""
import json
import multiprocessing as mp
import os
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from twcx.data import ZHANG_FREQS, load_zhang2020
from twcx.model import FadeModel, mae_pp, prepare, seconds_per_freq

ARMS = {"fixed_17.8Hz": ([30], False), "fixed_17.8+2.16Hz": ([30, 39], False),
        "full_sweep_ard": (list(range(60)), True)}


def run_fold(args):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    i, preps, cost = args
    test, train = preps[i], preps[:i] + preps[i + 1:]
    out = {"cell": test.name}
    for name, (idx, ard) in ARMS.items():
        t = time.time()
        m = FadeModel(idx, stride=4, ard=ard).fit(train)
        out[name] = {"mae_pp": mae_pp(m, test), "seconds": float(cost[idx].sum()),
                     "fit_s": round(time.time() - t, 1)}
    print("fold", test.name, {k: round(v["mae_pp"], 2) for k, v in out.items() if k != "cell"}, flush=True)
    return out


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    cells = load_zhang2020()
    preps = [prepare(c) for c in cells]
    cost = seconds_per_freq(ZHANG_FREQS)
    with mp.Pool(min(len(preps), os.cpu_count() or 1)) as pool:
        folds = pool.map(run_fold, [(i, preps, cost) for i in range(len(preps))])
    bench = json.load(open("results/benchmark.json"))
    plan2 = {f["cell"]: f["arms"]["twcx@2s"]["mae_pp"] for f in bench["folds"]}
    plan05 = {f["cell"]: f["arms"]["twcx@0.5s"]["mae_pp"] for f in bench["folds"]}
    extra = {}
    for name in ARMS:
        per = {f["cell"]: f[name]["mae_pp"] for f in folds}
        extra[name] = {"mean_mae_pp": float(np.mean(list(per.values()))), "per_cell": per,
                       "seconds": folds[0][name]["seconds"],
                       "planner_2s_wins_cells": sum(plan2[c] < per[c] for c in per),
                       "planner_0.5s_wins_cells": sum(plan05[c] < per[c] for c in per)}
    path = "results/compare_methods.json"
    res = json.load(open(path))
    res["extra"] = extra
    json.dump(res, open(path, "w"), indent=1)
    for k, v in extra.items():
        print(f"{k:<20}{v['mean_mae_pp']:7.2f} pp {v['seconds']:8.2f} s   planner 2 s wins "
              f"{v['planner_2s_wins_cells']}/7, 0.5 s wins {v['planner_0.5s_wins_cells']}/7")


if __name__ == "__main__":
    main()
