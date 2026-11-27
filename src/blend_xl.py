"""3-way fixed dev blends: user's EN_best_model + silver DeBERTa + XLM-R large (each / own threshold, blend thr 1.0)."""
import json
import numpy as np
from src.data import LABELS, ROOT, load_cv
from src.en import load_en
from src.final_model import load_dev
from src.post import _micro
from src.step10 import FINE, load_deberta, load_train_cv
cv = load_cv(); em = (cv.lang == "en").values
en, Y, F = load_en()
meta = json.load(open(ROOT / "ckpt/EN_best_model/meta.json"))
ext = np.load(ROOT / "runs/submissions/EN_best_model_EN_dev_scores.npy") / meta["threshold"]
db, dbd, _ = load_deberta("deb_silver_bce", Y, F); dbd = dbd / max(FINE, key=lambda t: _micro(Y, db >= t))
xl, _, _ = load_train_cv("phase1c/XL_E1", em, cv.fold.values, np.stack(cv.labels).astype(bool), cv.lang.values)
tx = max(FINE, key=lambda t: _micro(Y, xl >= t))
xld = np.load(ROOT / "runs/submissions/phase1c_XL_E1_EN_dev_scores.npy")[:, :14] / tx
best = (0.6 * ext + 0.4 * dbd) >= 1.0  # the 44.54 submission
print(f"XLM-R large threshold {tx}; XLM-R dev labels/para at its threshold {(xld >= 1).sum(1).mean():.2f}")
V = {"blend3_50_30_20xl": (0.5, 0.3, 0.2), "blend3_45_35_20xl": (0.45, 0.35, 0.2), "blend3_40_30_30xl": (0.4, 0.3, 0.3)}
dev = load_dev("en")
for n, (a, b, c) in V.items():
    Pd = (a * ext + b * dbd + c * xld) >= 1.0
    with open(ROOT / f"runs/submissions/{n}_EN_dev.txt", "w", encoding="utf-8", newline="\n") as f:
        for r, p in zip(dev.itertuples(), Pd):
            f.write("\t".join([r.doc, r.start, r.end] + [LABELS[i] for i in np.flatnonzero(p)]) + "\n")
    print(f"{n:20s} {Pd.sum(1).mean():.2f} labels/para, {100 * (Pd.sum(1) == 0).mean():4.1f}% None, "
          f"differs from the 44.54 blend on {(Pd != best).any(1).sum()} paragraphs")
# same 45/35/20 mix, threshold raised until labels/para matches the 44.54 blend (no dev labels used)
S = 0.45 * ext + 0.35 * dbd + 0.2 * xld; target = best.sum(1).mean()
T = min(np.round(np.arange(1.0, 1.5, 0.01), 2), key=lambda t: abs((S >= t).sum(1).mean() - target))
Pd = S >= T; n = "blend3_45_35_20xl_matched"
with open(ROOT / f"runs/submissions/{n}_EN_dev.txt", "w", encoding="utf-8", newline="\n") as f:
    for r, p in zip(dev.itertuples(), Pd):
        f.write("\t".join([r.doc, r.start, r.end] + [LABELS[i] for i in np.flatnonzero(p)]) + "\n")
print(f"{n}: thr {T}, {Pd.sum(1).mean():.2f} labels/para, {100 * (Pd.sum(1) == 0).mean():.1f}% None, differs from 44.54 on {(Pd != best).any(1).sum()}")
