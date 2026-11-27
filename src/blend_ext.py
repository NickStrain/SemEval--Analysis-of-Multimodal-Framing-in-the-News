"""Blend the user's EN_best_model with silver DeBERTa on the EN dev set (no new training).

Scale: each model's probabilities are divided by its own global threshold (1.0 = that model's decision boundary).
  user's model : its own threshold from meta.json (0.3)
  silver DeB   : threshold chosen on our EN CV
Weight w and final threshold T: chosen on our EN CV, using X1_mpnet (same encoder as the user's model, trained in our
folds) as the stand-in for the user's model. Reported nested (w, T picked without the test fold), then fit on all folds.
Writes runs/submissions/blend_EN_dev.txt (+ zip after the official format check) and EN_best_model_EN_dev.txt.
"""
import json

import numpy as np

from src.data import LABELS, ROOT
from src.en import per_fold
from src.final_model import load_dev
from src.post import _micro
from src.score_mcpt import F, Y, en_oof, nested_epoch
from src.step10 import FINE, load_deberta

meta = json.load(open(ROOT / "ckpt/EN_best_model/meta.json"))
ext_dev = np.load(ROOT / "runs/submissions/EN_best_model_EN_dev_scores.npy")
x1, _, ch = nested_epoch(en_oof("phase2m/X1_mpnet"))
db, db_dev, _ = load_deberta("deb_silver_bce", Y, F)
thr = lambda S, m: max(FINE, key=lambda t: _micro(Y[m], S[m] >= t))
W = np.round(np.arange(0, 1.01, 0.05), 2); TT = np.round(np.arange(0.5, 1.6, 0.02), 2)


def fit(m):
    tx, td = thr(x1, m), thr(db, m)
    best = max(((_micro(Y[m], (w * x1[m] / tx + (1 - w) * db[m] / td) >= T), w, T) for w in W for T in TT))
    return tx, td, best[1], best[2]


P = np.zeros(Y.shape, bool); Px = np.zeros(Y.shape, bool)
for k in range(5):
    tr, te = F != k, F == k
    tx, td, w, T = fit(tr)
    P[te] = (w * x1[te] / tx + (1 - w) * db[te] / td) >= T; Px[te] = x1[te] >= tx
for name, Q in [("mpnet (stand-in for your model) alone", Px), ("blend mpnet + silver DeBERTa", P)]:
    mi, ma = per_fold(Y, Q, F)
    print(f"CV EN  {name:40s} micro {100 * mi.mean():.2f} ± {100 * mi.std():.2f}  macro {100 * ma.mean():.1f}  folds {np.round(100 * mi, 1)}")
tx, td, w, T = fit(np.ones(len(Y), bool))
print(f"final: weight on your model {w}, DeBERTa {round(1 - w, 2)}, DeBERTa thr {td}, blend thr {T}")
S = w * ext_dev / meta["threshold"] + (1 - w) * db_dev / td
dev = load_dev("en")
for fname, Pd in [("blend_EN_dev.txt", S >= T), ("EN_best_model_EN_dev.txt", ext_dev >= meta["threshold"])]:
    path = ROOT / "runs/submissions" / fname
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r, p in zip(dev.itertuples(), Pd):
            f.write("\t".join([r.doc, r.start, r.end] + [LABELS[i] for i in np.flatnonzero(p)]) + "\n")
    print(f"wrote {path.name}: {Pd.sum(1).mean():.2f} labels/para, {100 * (Pd.sum(1) == 0).mean():.1f}% None")
Pb, Pe = S >= T, ext_dev >= meta["threshold"]
print(f"dev paragraphs where the blend differs from your model: {(Pb != Pe).any(1).sum()} of {len(Pb)}")
json.dump(dict(w_user=float(w), w_deberta=float(round(1 - w, 2)), thr_user=meta["threshold"], thr_deberta=float(td), thr_blend=float(T)),
          open(ROOT / "configs/blend_en.json", "w"), indent=1)
