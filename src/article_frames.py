"""Step 8a: predicted article frames from OUT-OF-FOLD ensemble probabilities (mCPT E1 0.4 + XLM-R base 0.6).
Article frames = union over its paragraphs of labels with prob >= tau; tau tuned (on OOF) for article-level micro-F1
against the gold union. Writes data/artframes.json {pid: "[frames: A, B] "} for all CV rows (all languages)."""
import json

import numpy as np

from src.data import LABELS, ROOT, load_cv

cv = load_cv(); Y = np.stack(cv.labels).astype(bool)
S = 0.4 * np.load(ROOT / "oof/mcpt_all.npy") + 0.6 * np.load(ROOT / "oof/xlmrb_all.npy")
arts = cv.article_id.values; ua = np.unique(arts)
G = np.stack([Y[arts == a].any(0) for a in ua])
mx = np.stack([S[arts == a].max(0) for a in ua])
def f1(P):
    tp = (P & G).sum(); return 2 * tp / max(1, P.sum() + G.sum())
taus = np.round(np.arange(0.1, 0.91, 0.05), 2); scores = [f1(mx >= t) for t in taus]
tau = float(taus[int(np.argmax(scores))])
print(f"article-level micro-F1 by tau: best {max(scores):.3f} at tau={tau}  (gold frames/article {G.sum(1).mean():.2f}, "
      f"pred {(mx >= tau).sum(1).mean():.2f})")
art_str = {a: "[frames: " + ", ".join(LABELS[j].split('_')[0] for j in np.flatnonzero(mx[i] >= tau)) + "] " for i, a in enumerate(ua)}
pid = cv.article_id + "#" + cv.para_idx.astype(str)
json.dump({p: art_str[a] for p, a in zip(pid, arts)}, open(ROOT / "data/artframes.json", "w"))
json.dump({"tau": tau, "weights": [0.4, 0.6]}, open(ROOT / "configs/artframes.json", "w"))
print("example:", art_str[ua[0]])
