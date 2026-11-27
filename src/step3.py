"""Step 3: error analysis of baseline mcpt_en OOF predictions (cross-fitted) -> runs/errors.md"""
import numpy as np
from transformers import AutoTokenizer

from src.data import LABELS, load_cv
from src.decode import crossfit
from src.en import load_en
from src.score import official_scores

cv = load_cv(); Yall = np.stack(cv.labels).astype(bool); em = (cv.lang == "en").values
z = np.load("runs/phase1b/M_E1_extraEN/oof.npz")["scores"]
Pall, _ = crossfit(z, Yall, cv.fold.values, cv.lang.values, "implicit", False)
en, Y, F = load_en(); P = Pall[em]
t = np.c_[Y, ~Y.any(1)]; p = np.c_[P, ~P.any(1)]
TP, FP, FN = (t & p).sum(0), (~t & p).sum(0), (t & ~p).sum(0)
tp, fp, fn = TP.sum(), FP.sum(), FN.sum(); base = 2 * tp / (2 * tp + fp + fn)
cost = [(2 * (tp + FN[j]) / (2 * (tp + FN[j]) + fp - FP[j] + fn - FN[j])) - base for j in range(15)]
L = ["# Step 3 — error analysis (mcpt_en, EN CV out-of-fold, official 15-class scoring)", "",
     f"Overall micro {100 * base:.2f}. *Micro cost* = micro-F1 gain if this label's FP and FN were all fixed.", "",
     "| label | TP | FP | FN | precision | recall | micro cost |", "|---|---|---|---|---|---|---|"]
for j, lab in sorted(enumerate(LABELS + ["None"]), key=lambda x: -cost[x[0]]):
    pr = TP[j] / max(1, TP[j] + FP[j]); rc = TP[j] / max(1, TP[j] + FN[j])
    L.append(f"| {lab} | {TP[j]} | {FP[j]} | {FN[j]} | {pr:.2f} | {rc:.2f} | +{100 * cost[j]:.2f} |")
gN, pN = ~Y.any(1), ~P.any(1)
L += ["", "## None errors", f"- predicted a frame when gold is None: **{(gN & ~pN).sum()}** of {gN.sum()} None paragraphs",
      f"- predicted None when gold has frames: **{(~gN & pN).sum()}** of {(~gN).sum()} framed paragraphs", ""]
tok = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
n = np.array([len(x) for x in tok(en.text.tolist(), add_special_tokens=False)["input_ids"]])
L += ["## Micro-F1 by paragraph length (tokens)", "", "| length | paragraphs | gold None % | micro |", "|---|---|---|---|"]
for name, m in [("< 40", n < 40), ("40-100", (n >= 40) & (n <= 100)), ("> 100", n > 100)]:
    L.append(f"| {name} | {m.sum()} | {100 * gN[m].mean():.0f}% | {100 * official_scores(Y[m], P[m])[0]:.1f} |")
top = sorted(range(15), key=lambda j: -cost[j])[:4]
L += ["", "## Top 4 labels by micro cost", ""] + [f"{i + 1}. {(LABELS + ['None'])[j]} (+{100 * cost[j]:.2f})" for i, j in enumerate(top)]
open("runs/errors.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
