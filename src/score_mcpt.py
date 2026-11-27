"""Score mCPT runs on ENGLISH only, honestly: for each test fold k the epoch and ONE global threshold are chosen on
the other 4 folds' English rows. Also reports the run in the best ensemble (with XLM-R large + silver DeBERTa).

Usage: python -m src.score_mcpt phase1b/M_E1_extraEN phase2m/S1_silver ...
"""
import sys

import numpy as np

from src.data import ROOT, load_cv
from src.en import load_en, per_fold
from src.ens import weight_grid
from src.post import _micro
from src.step10 import FINE, load_deberta, load_train_cv

cv = load_cv(); en_mask = (cv.lang == "en").values
en, Y, F = load_en()


def en_oof(run):
    """-> (C, N_en, 14) candidate scores (epochs, then stage-2 epochs if present)."""
    d = ROOT / "runs" / run
    c = [np.load(d / "oof.npz")["scores"][:, en_mask, :14]]
    if (d / "oof_stage2.npz").exists():
        c.append(np.load(d / "oof_stage2.npz")["scores"][:, en_mask, :14])
    z = np.concatenate(c)
    return z[~np.isnan(z).any((1, 2))]


def nested_epoch(z):
    oof = np.zeros(Y.shape, np.float32); P = np.zeros(Y.shape, bool); ch = []
    for k in range(5):
        tr, te = F != k, F == k
        best = max(((_micro(Y[tr], z[e][tr] >= t), e, t) for e in range(len(z)) for t in FINE[::2]))
        _, e, t = best; ch.append(e + 1); oof[te] = z[e][te]; P[te] = z[e][te] >= t
    return oof, P, ch


def nested_ens(oofs):
    P = np.zeros(Y.shape, bool)
    for k in range(5):
        tr, te = F != k, F == k; best = (-1, None, None)
        for w in weight_grid(len(oofs), 0.1):
            S = sum(a * o for a, o in zip(w, oofs))
            for t in FINE[::2]:
                f = _micro(Y[tr], S[tr] >= t)
                if f > best[0]:
                    best = (f, w, t)
        P[te] = sum(a * o[te] for a, o in zip(best[1], oofs)) >= best[2]
    return P


xl, _, _ = load_train_cv("phase1c/XL_E1", en_mask, cv.fold.values, np.stack(cv.labels).astype(bool), cv.lang.values)
db, _, _ = load_deberta("deb_silver_bce", Y, F)
for run in (sys.argv[1:] if __name__ == "__main__" else []):
    if not (ROOT / "runs" / run / "oof.npz").exists():
        print(f"{run}: not finished yet (oof.npz is written when all 5 folds are done) -> check: bash runs/track.sh"); continue
    oof, P, ch = nested_epoch(en_oof(run))
    mi, ma = per_fold(Y, P, F)
    mi3, ma3 = per_fold(Y, nested_ens([oof, xl, db]), F)
    print(f"{run:40s} alone {100 * mi.mean():.2f} ± {100 * mi.std():.2f} (macro {100 * ma.mean():.1f}) folds {np.round(100 * mi, 1)} "
          f"| + XLM-R L + silver DeBERTa {100 * mi3.mean():.2f} ± {100 * mi3.std():.2f}  | epochs {ch}", flush=True)
