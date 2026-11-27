"""No-training dev submissions: (1) user's EN_best_model alone, (2) fixed blend 0.7 user + 0.3 silver DeBERTa.
Each model's probabilities are divided by its own global threshold (user: meta.json; DeBERTa: tuned on our EN CV
OOF), blend threshold 1.0. The blend weights are a fixed judgement call (not validated)."""
import json
import numpy as np
from src.data import LABELS, ROOT
from src.en import load_en
from src.final_model import load_dev
from src.post import _micro
from src.step10 import FINE, load_deberta
en, Y, F = load_en()
meta = json.load(open(ROOT / "ckpt/EN_best_model/meta.json"))
ext = np.load(ROOT / "runs/submissions/EN_best_model_EN_dev_scores.npy")
db, db_dev, _ = load_deberta("deb_silver_bce", Y, F)
td = max(FINE, key=lambda t: _micro(Y, db >= t))
S = 0.7 * ext / meta["threshold"] + 0.3 * db_dev / td
dev = load_dev("en")
outs = {"EN_best_model_EN_dev.txt": ext >= meta["threshold"], "blend70_30_EN_dev.txt": S >= 1.0}
for fname, Pd in outs.items():
    with open(ROOT / "runs/submissions" / fname, "w", encoding="utf-8", newline="\n") as f:
        for r, p in zip(dev.itertuples(), Pd):
            f.write("\t".join([r.doc, r.start, r.end] + [LABELS[i] for i in np.flatnonzero(p)]) + "\n")
    print(f"{fname}: {Pd.sum(1).mean():.2f} labels/para, {100 * (Pd.sum(1) == 0).mean():.1f}% None")
a, b = outs.values()
print(f"DeBERTa threshold {td}; paragraphs where the blend differs from your model: {(a != b).any(1).sum()} of {len(a)}; "
      f"labels added {(b & ~a).sum()}, removed {(a & ~b).sum()}")
