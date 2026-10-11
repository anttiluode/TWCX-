"""Cost-aware frequency selection.

The question TWC's identifiability work kept arriving at, in its practical
form: given what the training cells show, which measurements are worth their
time? Selection sees ONLY the cells it is given. In the benchmark that means
the held-out cell never influences which frequencies are chosen.

Score of a frequency set S: leave-one-cell-out mean absolute SOH error over
the cells supplied (each cell is one fold, so a set has to work on cells it
was not fitted to). Greedy forward search under a per-test time budget.
"""
from __future__ import annotations

import numpy as np

from .model import FadeModel, mae_pp


class CVScorer:
    def __init__(self, preps, stride: int = 8, optimize: bool = True):
        self.preps = preps
        self.stride = stride
        self.optimize = optimize
        self.cache: dict = {}

    def __call__(self, idx) -> float:
        key = tuple(sorted(idx))
        if key not in self.cache:
            errs = []
            for i, p in enumerate(self.preps):
                tr = self.preps[:i] + self.preps[i + 1:]
                m = FadeModel(key, stride=self.stride, optimize=self.optimize).fit(tr)
                errs.append(mae_pp(m, p))
            self.cache[key] = float(np.mean(errs))
        return self.cache[key]


def greedy_budget(scorer: CVScorer, cost: np.ndarray, budget_s: float,
                  max_k: int = 3, min_rel_gain: float = 0.03):
    """Add the frequency that lowers CV error most while staying inside the
    budget; stop when the best addition improves CV error by less than
    `min_rel_gain` (relative) or nothing fits. Starts from the empty set,
    whose score is the measure-nothing baseline, so a frequency is only
    chosen if it beats guessing."""
    chosen: list[int] = []
    best = scorer(())
    history = [{"set": [], "cv_mae_pp": best, "seconds": 0.0}]
    while len(chosen) < max_k:
        spent = float(cost[chosen].sum()) if chosen else 0.0
        cands = [j for j in range(len(cost)) if j not in chosen and spent + cost[j] <= budget_s]
        if not cands:
            break
        scores = {j: scorer(chosen + [j]) for j in cands}
        j = min(scores, key=scores.get)
        if scores[j] > best * (1.0 - min_rel_gain):
            break
        chosen.append(j)
        best = scores[j]
        history.append({"set": list(chosen), "cv_mae_pp": best,
                        "seconds": float(cost[chosen].sum())})
    return chosen, history
