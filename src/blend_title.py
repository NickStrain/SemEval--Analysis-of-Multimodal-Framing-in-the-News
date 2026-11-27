"""The 45.07 submission (45/35/20 + thr 1.11) with the title rule: title paragraphs need score >= 3x the threshold.
Plus a variant where every title is None. No training; rule chosen on our CV / held-out 2b, not on dev."""
import json
import numpy as np
from src.data import LABELS, ROOT, load_cv
from src.en import load_en
from src.final_model import load_dev
from src.post import _micro
from src.step10 import FINE, load_deberta, load_train_cv
cv = load_cv(); em = (cv.lang == "en").values; en, Y, F = load_en()
om = json.load(open(ROOT / "ckpt/EN_best_model/meta.json")); orig = np.load(ROOT / "runs/submissions/EN_best_model_EN_dev_scores.npy") / om["threshold"]
db, dbd, _ = load_deberta("deb_silver_bce", Y, F); dbd = dbd / max(FINE, key=lambda t: _micro(Y, db >= t))
xl, _, _ = load_train_cv("phase1c/XL_E1", em, cv.fold.values, np.stack(cv.labels).astype(bool), cv.lang.values)
xld = np.load(ROOT / "runs/submissions/phase1c_XL_E1_EN_dev_scores.npy")[:, :14] / max(FINE, key=lambda t: _micro(Y, xl >= t))
S = 0.45 * orig + 0.35 * dbd + 0.2 * xld; T = 1.11; best = S >= T
dev = load_dev("en"); title = (dev.groupby("doc").cumcount() == 0).values
assert (dev.text[title] == dev.title[title]).all()
V = {"best45_titlerule3x": np.where(title[:, None], S >= 3 * T, S >= T), "best45_titles_none": best & ~title[:, None]}
for n, P in V.items():
    with open(ROOT / f"runs/submissions/{n}_EN_dev.txt", "w", encoding="utf-8", newline="\n") as f:
        for r, p in zip(dev.itertuples(), P):
            f.write("\t".join([r.doc, r.start, r.end] + [LABELS[i] for i in np.flatnonzero(p)]) + "\n")
    print(f"{n:22s} titles {title.sum()}: None among titles {(~best[title].any(1)).mean():.0%} -> {(~P[title].any(1)).mean():.0%}; "
          f"changed paragraphs vs 45.07: {(P != best).any(1).sum()}")
