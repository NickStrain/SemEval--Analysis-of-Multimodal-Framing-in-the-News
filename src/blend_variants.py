"""More fixed (untrained) dev blends around the 44.4 one. Each model / own threshold; mCPT thr from our EN CV."""
import json
import numpy as np
from src.data import LABELS, ROOT
from src.en import load_en
from src.final_model import load_dev
from src.post import _micro
from src.step10 import FINE, load_deberta
en, Y, F = load_en()
meta = json.load(open(ROOT / "ckpt/EN_best_model/meta.json"))
ext = np.load(ROOT / "runs/submissions/EN_best_model_EN_dev_scores.npy") / meta["threshold"]
db, dbd, _ = load_deberta("deb_silver_bce", Y, F); dbd = dbd / max(FINE, key=lambda t: _micro(Y, db >= t))
mc = np.load(ROOT / "oof/mcpt_en.npy"); mcd = np.load(ROOT / "dev/mcpt_en.npy")[:, :14] / max(FINE, key=lambda t: _micro(Y, mc >= t))
base = (0.7 * ext + 0.3 * dbd) >= 1.0
V = {"blend60_40": (0.6 * ext + 0.4 * dbd, 1.0), "blend80_20": (0.8 * ext + 0.2 * dbd, 1.0),
     "blend70_30_thr105": (0.7 * ext + 0.3 * dbd, 1.05),
     "blend3_60_25_15mcpt": (0.6 * ext + 0.25 * dbd + 0.15 * mcd, 1.0)}
dev = load_dev("en")
for n, (S, T) in V.items():
    Pd = S >= T
    with open(ROOT / f"runs/submissions/{n}_EN_dev.txt", "w", encoding="utf-8", newline="\n") as f:
        for r, p in zip(dev.itertuples(), Pd):
            f.write("\t".join([r.doc, r.start, r.end] + [LABELS[i] for i in np.flatnonzero(p)]) + "\n")
    print(f"{n:22s} {Pd.sum(1).mean():.2f} labels/para, {100 * (Pd.sum(1) == 0).mean():4.1f}% None, "
          f"differs from the 44.4 blend on {(Pd != base).any(1).sum()} paragraphs")
