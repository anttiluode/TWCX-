"""Offline tests: synthetic cells with a known answer. No network."""
import csv
import os
import tempfile

import numpy as np

from twcx.data import Cell, load_long_csv
from twcx.model import FadeModel, mae_pp, prepare, seconds_per_freq
from twcx.select import CVScorer, greedy_budget

FREQS = np.array([1000.0, 100.0, 10.0, 1.0, 0.1])


def synth_cells(n_cells=5, n_tests=60, signal_idx=2, seed=0):
    """SOH fades linearly; only -Im at FREQS[signal_idx] tracks it."""
    rng = np.random.default_rng(seed)
    cells = []
    for c in range(n_cells):
        soh = 1.0 - np.linspace(0, 0.3 + 0.05 * c, n_tests)
        cap0 = 40 + rng.normal(0, 2)
        re = 0.5 + rng.normal(0, 0.02, (n_tests, len(FREQS)))
        im = 0.1 + rng.normal(0, 0.02, (n_tests, len(FREQS)))
        im[:, signal_idx] += 2.0 * (1.0 - soh)
        cells.append(Cell(f"c{c}", FREQS.copy(), re, im, cap0 * soh))
    return cells


def test_cost_is_monotone_in_period():
    t = seconds_per_freq(FREQS)
    assert np.all(np.diff(t) > 0)
    assert abs(t[-1] - (3 / 0.1 + 0.2)) < 1e-12


def test_prepare_uses_commissioning_reference():
    p = prepare(synth_cells(1)[0], k0=5)
    assert abs(p.soh[:5].mean() - 1.0) < 1e-12
    assert not p.eval_mask[:5].any() and p.eval_mask[5:].all()
    assert np.allclose(p.d_im[:5].mean(0), 0.0)


def test_model_learns_signal_and_nothing_baseline_does_not():
    preps = [prepare(c) for c in synth_cells()]
    good = np.mean([mae_pp(FadeModel([2]).fit(preps[:i] + preps[i + 1:]), preps[i]) for i in range(5)])
    none = np.mean([mae_pp(FadeModel([]).fit(preps[:i] + preps[i + 1:]), preps[i]) for i in range(5)])
    assert good < 0.3 * none


def test_greedy_finds_informative_frequency_within_budget():
    preps = [prepare(c) for c in synth_cells(signal_idx=2)]
    cost = seconds_per_freq(FREQS)
    idx, hist = greedy_budget(CVScorer(preps, stride=2), cost, budget_s=1.0)
    assert idx and idx[0] == 2
    assert cost[idx].sum() <= 1.0


def test_greedy_respects_budget_that_excludes_signal():
    preps = [prepare(c) for c in synth_cells(signal_idx=4)]   # signal only at 0.1 Hz (30.2 s)
    cost = seconds_per_freq(FREQS)
    idx, _ = greedy_budget(CVScorer(preps, stride=2), cost, budget_s=1.0)
    assert 4 not in idx
    assert cost[idx].sum() <= 1.0 if idx else True


def test_long_csv_roundtrip():
    cells = synth_cells(3, n_tests=8)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cell", "test", "freq_hz", "z_re", "z_im", "capacity"])
            for c in cells:
                for t in range(c.n):
                    for k, fz in enumerate(c.freqs):
                        w.writerow([c.name, t, fz, c.re[t, k], -c.neg_im[t, k], c.capacity[t]])
        back = load_long_csv(path)
    assert [b.name for b in back] == [c.name for c in cells]
    for b, c in zip(back, cells):
        assert np.allclose(b.freqs, c.freqs)
        assert np.allclose(b.re, c.re) and np.allclose(b.neg_im, c.neg_im)
        assert np.allclose(b.capacity, c.capacity)
