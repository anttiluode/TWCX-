"""Inject results/benchmark.json + results/frontier_zhang.json into the
workbench page. Every number on docs/workbench.html comes from those files.

  python docs/build_workbench.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bench = json.load(open(os.path.join(ROOT, "results", "benchmark.json")))
front = json.load(open(os.path.join(ROOT, "results", "frontier_zhang.json")))

budgets = [f"{b:g}" for b in bench["budgets_s"]]
arms = {}
for k, s in bench["summary"].items():
    arms[k] = {"mae": s["mean_mae_pp"], "secs": s["mean_seconds_per_test"],
               "per": s["per_cell_mae_pp"], "eol": s.get("eol_flag")}

folds = []
for f in bench["folds"]:
    folds.append({"cell": f["cell"],
                  "sel2": f["arms"]["twcx@2s"]["freqs_hz"],
                  "sel05": f["arms"]["twcx@0.5s"]["freqs_hz"]})
picks05 = [x for f in folds for x in f["sel05"]]
band = [min(picks05), max(picks05)]

plans, seen = [], set()
for p in front["plans"]:
    key = tuple(round(x, 4) for x in p["freqs_hz"])
    if key in seen:
        continue
    seen.add(key)
    b = p["budget_s"]
    plans.append({"budget_s": b, "budget_key": f"{b:g}", "budget_label": f"{b:g} s",
                  "freqs_hz": p["freqs_hz"], "seconds_per_test": p["seconds_per_test"],
                  "cv_mae_pp": p["cv_mae_pp"]})
# the last distinct plan also covers every larger budget
if plans:
    plans[-1]["budget_label"] = f"{plans[-1]['budget_s']:g} s and up"

data = {
    "arms": arms, "budgets": budgets, "folds": folds, "band": band,
    "bandDecades": math.log10(band[1] / band[0]),
    "lowest2": min(x for f in folds for x in f["sel2"]),
    "cells": bench["cells"], "plans": plans,
    "fullSweepSeconds": bench["full_sweep_seconds"],
    "costModel": front["cost_model"],
}
tpl = open(os.path.join(ROOT, "docs", "workbench.template.html")).read()
out = tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
open(os.path.join(ROOT, "docs", "workbench.html"), "w").write(out)
print("wrote docs/workbench.html", len(out), "bytes")

# Standalone copy for GitHub Pages (the artifact host adds this skeleton itself).
page = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<style>body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style>'
        '</head><body>\n' + out + '\n</body></html>\n')
open(os.path.join(ROOT, "index.html"), "w").write(page)
print("wrote index.html (GitHub Pages)")
