"""Best 45.07 blend recipe (45/35/20 + label-rate-matched threshold) with the fine-tuned copy (ft1 best_e3) in
place of the original user model. Also a 2-way 60/40 version. No training."""
import json
import numpy as np
from src.data import LABELS, ROOT, load_cv
from src.en import load_en
from src.final_model import load_dev
from src.post import _micro
from src.step10 import FINE, load_deberta, load_train_cv
cv = load_cv(); em = (cv.lang == "en").values; en, Y, F = load_en()
d = ROOT / "ckpt/EN_best_model_ft/ft1/best_e3"; m = json.load(open(d / "meta.json"))
ft = np.load(d / "dev_scores.npy") / m["threshold"]
om = json.load(open(ROOT / "ckpt/EN_best_model/meta.json")); orig = np.load(ROOT / "runs/submissions/EN_best_model_EN_dev_scores.npy") / om["threshold"]
db, dbd, _ = load_deberta("deb_silver_bce", Y, F); dbd = dbd / max(FINE, key=lambda t: _micro(Y, db >= t))
xl, _, _ = load_train_cv("phase1c/XL_E1", em, cv.fold.values, np.stack(cv.labels).astype(bool), cv.lang.values)
xld = np.load(ROOT / "runs/submissions/phase1c_XL_E1_EN_dev_scores.npy")[:, :14] / max(FINE, key=lambda t: _micro(Y, xl >= t))
Sb = 0.45 * orig + 0.35 * dbd + 0.2 * xld; best = Sb >= 1.11           # the 45.07 submission
dev = load_dev("en"); grid = np.round(np.arange(0.9, 1.5, 0.01), 2)
def write(n, P):
    with open(ROOT / f"runs/submissions/{n}_EN_dev.txt", "w", encoding="utf-8", newline="\n") as f:
        for r, p in zip(dev.itertuples(), P):
            f.write("\t".join([r.doc, r.start, r.end] + [LABELS[i] for i in np.flatnonzero(p)]) + "\n")
    print(f"{n:28s} {P.sum(1).mean():.2f} labels/para, {100 * (P.sum(1) == 0).mean():4.1f}% None, differs from 45.07 on {(P != best).any(1).sum()}")
print(f"ft e3 alone at its thr: {(ft >= 1).sum(1).mean():.2f} labels/para, {100 * ((ft >= 1).sum(1) == 0).mean():.1f}% None")
S = 0.45 * ft + 0.35 * dbd + 0.2 * xld
write("ft3_blend3_matched", S >= min(grid, key=lambda t: abs((S >= t).sum(1).mean() - best.sum(1).mean())))
S2 = 0.6 * ft + 0.4 * dbd
write("ft3_blend60_40", S2 >= 1.0)
