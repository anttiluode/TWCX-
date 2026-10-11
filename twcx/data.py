"""Data access.

Two sources:

1. `load_zhang2020()` — the real laboratory data published with
   Zhang, Tang, Zhang, Chen, Kang, Lee, "Identifying degradation patterns of
   lithium ion batteries from impedance spectroscopy using machine learning",
   Nature Communications 11, 1706 (2020). The preprocessed EIS + capacity
   tables ship inside the authors' own GitHub repository
   (YunweiZhang/ML-identify-battery-degradation, Code-Matlab.zip). The data is
   NOT vendored into this repo (no licence is stated on it); it is downloaded
   on first use, checked against a pinned SHA-256, and cached.

2. `load_long_csv(path)` — your own measurements, one row per
   (cell, test, frequency). See README for the column names.

Every loader returns a list of `Cell` objects with the same layout, so the
selector and the CLI never care where the data came from.
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass, field

import numpy as np

ZHANG_URL = ("https://raw.githubusercontent.com/YunweiZhang/"
             "ML-identify-battery-degradation/master/Code-Matlab.zip")
ZHANG_SHA256 = "7d6ab9b3d48a2eadf3a4a737d8c6c8c35a81f0f1d2780249cbb0e735f5e0f69c"

# Nominal 60-point grid. The preprocessed tables carry no frequency column.
# The paper's sweep is 20 kHz -> 0.02 Hz; a log grid with 60 points puts
# index 30 at 17.79 Hz and index 13 at 952.8 Hz. Treat labels as nominal.
ZHANG_FREQS = 20000.0 * 10 ** (-6.0 * np.arange(60) / 59.0)

# Cell boundaries inside EIS_data.txt (training file, state V, mixed
# temperatures). Found as the five upward capacity jumps > 5 mAh; asserted
# on load so a changed upstream file fails loudly instead of silently.
_ZHANG_BOUNDS = [0, 200, 450, 679, 760, 1059, 1358]
_ZHANG_NAMES = ["25C_A", "25C_B", "25C_C", "25C_D", "35C01", "45C01"]
_ZHANG_TEMPS = [25, 25, 25, 25, 35, 45]


@dataclass
class Cell:
    name: str
    freqs: np.ndarray          # (F,) Hz, descending
    re: np.ndarray             # (n_tests, F) Ohm
    neg_im: np.ndarray         # (n_tests, F) Ohm, -Im(Z)
    capacity: np.ndarray       # (n_tests,) measured capacity
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.capacity)


def _cache_dir() -> str:
    d = os.environ.get("TWCX_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "twcx"))
    os.makedirs(d, exist_ok=True)
    return d


def fetch_zhang_zip() -> bytes:
    path = os.path.join(_cache_dir(), "zhang2020_Code-Matlab.zip")
    if not os.path.exists(path):
        with urllib.request.urlopen(ZHANG_URL, timeout=300) as r:
            blob = r.read()
        with open(path, "wb") as f:
            f.write(blob)
    with open(path, "rb") as f:
        blob = f.read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != ZHANG_SHA256:
        raise RuntimeError(f"Zhang 2020 archive hash mismatch: {digest} != {ZHANG_SHA256}. "
                           f"Delete {path} and retry; if it persists the upstream file changed.")
    return blob


def _txt(z: zipfile.ZipFile, name: str) -> np.ndarray:
    with z.open("Code-Matlab/" + name) as f:
        return np.loadtxt(io.TextIOWrapper(f, encoding="ascii"))


def load_zhang2020() -> list[Cell]:
    z = zipfile.ZipFile(io.BytesIO(fetch_zhang_zip()))
    X = _txt(z, "EIS_data.txt")            # (1358, 120): 60 Re, then 60 -Im
    y = _txt(z, "Capacity_data.txt")       # (1358,) mAh
    X2 = _txt(z, "EIS_data_35C02.txt")     # held-out cell used in the paper
    y2 = _txt(z, "capacity35C02.txt")

    jumps = list(np.where(np.diff(y) > 5.0)[0] + 1)
    if [0] + jumps + [len(y)] != _ZHANG_BOUNDS:
        raise RuntimeError(f"unexpected cell boundaries {jumps}")

    cells = []
    for name, temp, a, b in zip(_ZHANG_NAMES, _ZHANG_TEMPS, _ZHANG_BOUNDS[:-1], _ZHANG_BOUNDS[1:]):
        cells.append(Cell(name, ZHANG_FREQS.copy(), X[a:b, :60], X[a:b, 60:], y[a:b],
                          {"temperature_C": temp, "source": "Zhang2020 EIS_data.txt rows %d-%d" % (a, b)}))
    cells.append(Cell("35C02", ZHANG_FREQS.copy(), X2[:, :60], X2[:, 60:], y2,
                      {"temperature_C": 35, "source": "Zhang2020 EIS_data_35C02.txt"}))
    return cells


def load_long_csv(path: str) -> list[Cell]:
    """Columns: cell, test, freq_hz, z_re, z_im, capacity.

    z_im is the ordinary imaginary part (negative for capacitive behaviour).
    `capacity` may repeat on every frequency row of a test; the first value
    is used. Every test of every cell must contain the same frequency set.
    """
    rows: dict = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["cell"], int(float(r["test"])))
            d = rows.setdefault(key, {"cap": float(r["capacity"]), "pts": {}})
            d["pts"][float(r["freq_hz"])] = (float(r["z_re"]), -float(r["z_im"]))
    freqs = None
    by_cell: dict = {}
    for (cell, test), d in sorted(rows.items()):
        fs = np.array(sorted(d["pts"], reverse=True))
        if freqs is None:
            freqs = fs
        elif len(fs) != len(freqs) or not np.allclose(fs, freqs, rtol=1e-3):
            raise ValueError(f"cell {cell} test {test}: frequency set differs from the first test")
        by_cell.setdefault(cell, []).append(
            (np.array([d["pts"][f][0] for f in fs]), np.array([d["pts"][f][1] for f in fs]), d["cap"]))
    out = []
    for cell, tests in by_cell.items():
        out.append(Cell(cell, freqs.copy(), np.array([t[0] for t in tests]),
                        np.array([t[1] for t in tests]), np.array([t[2] for t in tests]),
                        {"source": os.path.basename(path)}))
    return out
