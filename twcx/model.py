"""Measurement cost, features and the capacity-fade model.

Task (monitoring, not cold grading): each cell has a commissioning reference
— the mean of its first K0 impedance spectra and capacity checks. Later
spectra are expressed as CHANGE from that reference, and the model predicts
relative capacity r = C / C_ref (state of health, in percent when x100).

Why this framing: on the Zhang 2020 cells, impedance predicts ABSOLUTE
capacity of an unseen cell only about as well as guessing the training mean
(~3 mAh error on ~35 mAh cells either way; see "absolute_capacity_check" in
results/benchmark.json). Cell-to-cell spread in initial capacity is largely
invisible to EIS there. Fade relative to a cell's own reference is
predictable, so that is the job the tool is benchmarked on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

K0 = 5  # commissioning tests that define each cell's reference


def seconds_per_freq(freqs, settle_periods=1.0, measure_periods=2.0, overhead_s=0.2):
    """Estimated wall time to measure each frequency once.

    (settle + measured periods) / f + fixed overhead. Real potentiostats
    differ (integration cycles, multisine, drift checks); this is a stated,
    adjustable model — the frequency RANKING by cost is what matters, and
    that is set by 1/f, which no instrument escapes.
    """
    f = np.asarray(freqs, float)
    return (settle_periods + measure_periods) / f + overhead_s


@dataclass
class Prepared:
    name: str
    d_re: np.ndarray       # change of Re(Z) from reference, (n, F)
    d_im: np.ndarray       # change of -Im(Z) from reference, (n, F)
    soh: np.ndarray        # C / C_ref
    eval_mask: np.ndarray  # tests after commissioning


def prepare(cell, k0: int = K0) -> Prepared:
    if cell.n <= k0 + 1:
        raise ValueError(f"cell {cell.name} has only {cell.n} tests; need > {k0 + 1}")
    ref_re = cell.re[:k0].mean(0)
    ref_im = cell.neg_im[:k0].mean(0)
    soh = cell.capacity / cell.capacity[:k0].mean()
    mask = np.arange(cell.n) >= k0
    return Prepared(cell.name, cell.re - ref_re, cell.neg_im - ref_im, soh, mask)


def features(p: Prepared, idx) -> np.ndarray:
    idx = list(idx)
    return np.hstack([p.d_re[:, idx], p.d_im[:, idx]])


class FadeModel:
    """GP regression on standardized impedance changes. Empty idx = predict
    the training mean (the 'measure nothing' baseline)."""

    def __init__(self, idx, stride: int = 4, optimize: bool = True, seed: int = 0, ard: bool = False):
        self.idx = list(idx)
        self.stride = stride
        self.optimize = optimize
        self.seed = seed
        self.ard = ard

    def fit(self, train: list[Prepared]):
        y = np.hstack([p.soh[p.eval_mask][:: self.stride] for p in train])
        if not self.idx:
            self.mean_ = float(y.mean())
            self.std_ = float(y.std())
            return self
        X = np.vstack([features(p, self.idx)[p.eval_mask][:: self.stride] for p in train])
        self.sc_ = StandardScaler().fit(X)
        ls = np.ones(X.shape[1]) if self.ard else 1.0   # ARD: one lengthscale per feature
        k = (ConstantKernel(1.0, (1e-2, 1e2)) * RBF(ls, (1e-2, 1e4 if self.ard else 1e3))
             + WhiteKernel(0.05, (1e-5, 1.0)))
        self.gp_ = GaussianProcessRegressor(k, normalize_y=True,
                                            optimizer="fmin_l_bfgs_b" if self.optimize else None,
                                            random_state=self.seed)
        self.gp_.fit(self.sc_.transform(X), y)
        return self

    def predict(self, p: Prepared, return_std=False):
        n = len(p.soh)
        if not self.idx:
            m, s = np.full(n, self.mean_), np.full(n, self.std_)
        else:
            m, s = self.gp_.predict(self.sc_.transform(features(p, self.idx)), return_std=True)
        return (m, s) if return_std else m


def mae_pp(model: FadeModel, p: Prepared) -> float:
    """Mean absolute SOH error in percentage points on post-commissioning tests."""
    pred = model.predict(p)
    return float(np.mean(np.abs(pred - p.soh)[p.eval_mask]) * 100.0)
