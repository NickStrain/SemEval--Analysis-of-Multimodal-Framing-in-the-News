"""Pick BCE vs ASL on fold 0: best micro over epochs with fold-0-tuned thresholds (same optimistic protocol for both)."""
import numpy as np

from src.decode import decode, tune
from src.en import load_en
from src.score import official_scores

en, Y, F = load_en(); m = F == 0; best = {}
for name in ("deb_bce", "deb_asl"):
    z = np.load(f"runs/step4/{name}/scores.npz")["oof"]
    res = []
    for e in range(z.shape[0]):
        S = np.c_[z[e][m], np.zeros(m.sum())]
        T = tune(S, Y[m], "implicit"); res.append(official_scores(Y[m], decode(S, T, "implicit"))[0])
    best[name] = max(res); print(name, "fold-0 micro by epoch:", " ".join(f"{100 * r:.1f}" for r in res))
w = max(best, key=best.get); print("WINNER", w); open("runs/step4/winner.txt", "w").write(w)
