# TWCX — impedance test planner

Pick the impedance frequencies worth their test time, and get an honest estimate of how well they work on cells the planner never saw.

This is the practical residue of [TransientWaveCompiler](https://github.com/anttiluode/TransientWaveCompiler)'s identifiability work ("which measurement would resolve the ambiguity?"), pointed at a job people pay for: electrochemical impedance spectroscopy (EIS) for battery state-of-health, where low frequencies take minutes per test.

## Result (real lab data, held out)

Data: Zhang et al., *Nature Communications* 11, 1706 (2020). Seven Eunicell LR2032 coin cells, cycled at 25/35/45 °C, EIS at 60 frequencies from 20 kHz to 0.02 Hz, capacity measured alongside. Protocol: leave one cell out; the planner chooses frequencies from the other six only; seven folds.

Task: track capacity fade against each cell's own commissioning reference (mean of its first five tests). Error is mean absolute SOH error in percentage points.

| method | held-out error | est. time per test | 80 % flag, balanced accuracy |
|---|---:|---:|---:|
| measure nothing (guess the training mean) | 6.18 pp | 0 s | 0.500 |
| 1 kHz reading (953 Hz grid point) | 6.04 pp | 0.20 s | 0.609 |
| full 60-point sweep | 4.14 pp | 731 s | 0.762 |
| **planner, ≤ 0.5 s budget** | **3.05 pp** | 0.34 s | **0.832** |
| **planner, ≤ 2 s budget** | **2.65 pp** | 0.87 s | 0.794 |
| one random affordable frequency, ≤ 2 s | 5.03 pp | 0.40 s | – |
| best single frequency in hindsight (not achievable) | 2.03 pp | 0.47 s | – |

Cell-level, the 2 s plan beats measuring nothing on 7/7 cells, the 1 kHz reading on 7/7, and the full sweep on 6/7. The run is deterministic; a second full run reproduced `results/benchmark.json` exactly.

In plain terms: on these cells, under a second of mid-frequency measurement tracked fade better than a twelve-minute full sweep, and the 1 kHz reading that fast cell testers use carried essentially no fade information.

The interactive version, with the error-vs-time chart, the frequency map and a copyable test plan, is `docs/workbench.html` (built from the result files by `docs/build_workbench.py`).

## Against existing selection methods

`compare_methods.py` keeps everything identical (folds, features, final GP, budgets, at most 3 frequencies) and swaps only the rule that picks the frequencies. Every rule sees only the six training cells of its fold.

| selection rule | ≤ 0.5 s | ≤ 2 s |
|---|---:|---:|
| greedy, inner leave-one-cell-out (this repo) | 3.05 pp | 2.65 pp |
| rank by correlation with SOH | 3.50 pp | 3.24 pp |
| LASSO path | 3.50 pp | 2.99 pp |
| ARD-GP lengthscale ranking (the approach Zhang et al. used) | 4.92 pp | 2.64 pp |

`compare_extra.py` adds the baselines that need no selection at all, and a fairer full-sweep model:

| fixed choice | error | est. time per test | planner ≤ 2 s better on |
|---|---:|---:|---:|
| 17.8 Hz alone (the frequency Zhang et al. flag) | 3.03 pp | 0.37 s | 6 of 7 cells |
| 17.8 + 2.16 Hz (the published pair) | 3.46 pp | 1.96 s | 5 of 7 cells |
| full sweep, ARD kernel (one lengthscale per feature) | 3.52 pp | 731 s | 6 of 7 cells |

The ARD kernel improves the full-sweep model from 4.14 to 3.52 pp, so the headline gap was partly a weak baseline. The short plan still beats the full sweep on 6 of 7 cells. The single 17.8 Hz reading from the literature matches the planner's 0.5 s budget (3.03 against 3.05 pp) with no selection machinery at all.

Read cell by cell, the comparison with selection rules is not a win for the greedy rule. At 0.5 s, correlation and LASSO both pick 11.1 Hz in every fold and beat greedy on 5 of 7 cells; greedy's lower mean comes from two cells. At 2 s, greedy and ARD tie (2.65 vs 2.64). Greedy is clearly better than ARD at 0.5 s and than correlation at 2 s, and that is all.

The original paper's own pipeline (raw spectra, all 60 frequencies, isotropic GP), run on its showcase cell 35C02 and scored the way its Fig. 3a plots it, gives 1.06 pp. The planner's 2 s plan gives 1.38 pp on that cell. The scoring differs slightly (the paper normalises by the first estimate, this repo by a five-test reference), and 35C02 is an easy cell because its twin 35C01 is in training, but the original method does better there.

**Bottom line:** the finding that matters for test time, mid-frequency points instead of a full sweep or a 1 kHz reading, holds under every selection rule tried and against an ARD full-sweep model. Simple existing rules find it too. The selector in this repo is not a better algorithm; the repo's value is the packaged tool, the cost model and the honest held-out benchmark.

## What this does not show

- **Seven coin cells, one chemistry, one lab.** Enough to see a large effect; not enough to qualify a plan for a production line. Run the planner on your own cells.
- **Fade tracking, not cold grading.** Predicting the absolute capacity of an unknown cell from its spectrum came out about 3 mAh wrong on ~35 mAh cells whatever was measured, close to guessing the average (`absolute_capacity_check` in the results). The tool is for cells with a commissioning reference: aging labs, fleet or storage monitoring, on-board impedance.
- **The frequencies are not a discovery.** The original authors' own code (`ARD_GPR.m`) singles out the 91st feature, the imaginary part at grid index 30 (17.8 Hz nominal), as the top predictor. Picking a few informative EIS frequencies is an established research topic. What this repo adds is a reusable, cost-aware, nested held-out way of choosing the plan, and the direct comparison against the full sweep and 1 kHz. The selection rule itself does no better than simple existing rules (see above).
- **Frequency labels are nominal.** The published tables have no frequency column; the grid is reconstructed as 60 log-spaced points from 20 kHz to 0.02 Hz, which puts index 30 at 17.79 Hz.
- **Times are estimates.** Cost model: (1 settling + 2 measured periods)/f + 0.2 s per frequency. Instruments differ; the 1/f scaling that makes low frequencies expensive does not.
- **More budget did not help.** At 60 s one fold picked a 0.05 Hz point that fit its six training cells and failed on the held-out one. With six training cells, the band the planner picks is stable (all seven 0.5 s picks fall in 17.8–35.9 Hz); the exact points are not.
- **Adaptive per-test stopping was not repeated here.** An earlier attempt (Sol, another session) found adaptive measurement did not beat a fixed plan; this repo ships fixed plans only.

## Use it on your own cells

```
pip install numpy scipy scikit-learn
python -m twcx plan your_eis.csv --budget 2 --out plan.json
python -m twcx frontier your_eis.csv            # error vs time across budgets
python -m twcx plan --zhang --budget 2          # the public demo data
```

CSV, one row per (cell, test, frequency):

```
cell,test,freq_hz,z_re,z_im,capacity
A1,0,20000,0.385,0.035,37.2
A1,0,15825,0.392,0.017,37.2
...
```

`z_im` is the ordinary imaginary part (negative when capacitive). Every test must use the same frequency set. You need at least 3 cells; with fewer than about 5, treat the output as a shortlist to verify. Options: `--k0` commissioning tests per cell (default 5), `--settle`, `--periods`, `--overhead` for the time model.

The plan's printed selection score is optimistic, because the same cells chose the plan. `benchmark.py` shows how to get the honest number: nested leave-one-cell-out, as above.

## Reproduce

```
python benchmark.py                 # ~18 min on 2 cores, writes results/benchmark.json
python -m twcx frontier --zhang --out results/frontier_zhang.json
python compare_methods.py           # ~10 min, writes results/compare_methods.json
python compare_extra.py             # ~25 min (ARD on 120 features), adds "extra" to it
python docs/build_workbench.py      # rebuilds docs/workbench.html from those two files
python -m pytest -q tests           # offline tests on synthetic cells with a known answer
```

The demo data is downloaded on first use from the authors' repository ([YunweiZhang/ML-identify-battery-degradation](https://github.com/YunweiZhang/ML-identify-battery-degradation), `Code-Matlab.zip`), checked against a pinned SHA-256 and cached in `~/.cache/twcx`. It is not vendored here because no licence is stated on it.

## Layout

- `twcx/data.py` — demo-data fetch and cell segmentation, CSV loader
- `twcx/model.py` — time-cost model, commissioning-referenced features, GP fade model
- `twcx/select.py` — leave-one-cell-out scorer, greedy selection under a time budget
- `twcx/cli.py` — `plan` and `frontier`
- `benchmark.py` — the held-out benchmark above
- `docs/` — workbench page and its builder
