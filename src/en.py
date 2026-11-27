"""English-only CV helpers shared by Steps 1-10.

Rows = English CV paragraphs in load_cv() order (final split excluded). Folds = the existing 5 article-level
folds restricted to English (data/folds.json). OOF/dev probability files are (N,14) float32 in this row order.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from src.data import ROOT, load_cv
from src.score import official_scores

FOLDS_FILE = ROOT / "data" / "folds.json"
LOG = ROOT / "runs" / "log.csv"


def load_en():
    cv = load_cv()
    en = cv[cv.lang == "en"].reset_index(drop=True)
    en["pid"] = en.article_id + "#" + en.para_idx.astype(str)
    if not FOLDS_FILE.exists():
        FOLDS_FILE.write_text(json.dumps({str(k): sorted(en.article_id[en.fold == k].unique()) for k in range(5)}, indent=1))
    Y = np.stack(en.labels).astype(bool)
    return en, Y, en.fold.values


def per_fold(Y, P, folds):
    r = np.array([official_scores(Y[folds == k], P[folds == k]) for k in range(5)])
    return r[:, 0], r[:, 1]  # micro per fold, macro per fold


def summary(mi, ma):
    return f"micro {100 * mi.mean():.2f} ± {100 * mi.std():.2f}   macro {100 * ma.mean():.2f} ± {100 * ma.std():.2f}"


def keep(new_mi, old_mi):
    d = np.asarray(new_mi) - np.asarray(old_mi)
    ok = d.mean() >= 0.010 and d.mean() > d.std()
    return ok, f"Δmicro {100 * d.mean():+.2f} (paired std {100 * d.std():.2f}, better in {(d > 0).sum()}/5) -> {'KEEP' if ok else 'not kept'}"


def log_run(step, model, config, mi, ma, dev_micro=""):
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "step", "model", "config", "cv_micro", "cv_micro_std", "cv_macro", "dev_micro"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), step, model, config, f"{100 * mi.mean():.2f}",
                    f"{100 * mi.std():.2f}", f"{100 * ma.mean():.2f}", dev_micro])
