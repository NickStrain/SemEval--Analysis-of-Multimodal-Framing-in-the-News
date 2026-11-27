"""English-only F1 for every run (official 15-class scoring on EN paragraphs only).

  global thr : thresholds/checkpoint cross-fitted on all languages of the other folds, applied to EN
  EN thr     : thresholds/checkpoint cross-fitted on the other folds' EN paragraphs only
Usage: python -m src.en_report      -> runs/en_report.md
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from src.data import LABELS, ROOT, load_cv
from src.decode import crossfit, with_none
from src.eval import official_scores

df = load_cv(); Y = np.stack(df.labels).astype(bool); F, L = df.fold.values, df.lang.values; EN = L == "en"
R = ROOT / "runs"
RUNS = [(p.parent.parent.name + "/" + p.parent.name + ("" if p.name == "oof.npz" else " [stage2]"), p)
        for p in sorted(R.glob("phase1*/*/oof*.npz")) if "Z_prior" not in str(p)]


def score(item):
    name, path = item
    S = np.load(path)["scores"]; out = {}
    for mode in ("implicit", "explicit"):
        if np.isnan(S[:, EN, 14]).all() and mode == "explicit":
            continue
        for scope in ("global", "EN"):
            if scope == "global":
                if np.isnan(S[:, ~EN, 0]).all():
                    continue
                pred, _ = crossfit(S, Y, F, L, mode, False); p, y, f = pred[EN], Y[EN], F[EN]
            else:
                p, _ = crossfit(S[:, EN], Y[EN], F[EN], L[EN], mode, False); y, f = Y[EN], F[EN]
            per = [official_scores(y[f == k], p[f == k]) for k in range(5)]
            out[(mode, scope)] = (np.array(per), p)
    return name, out


if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex:
        res = dict(ex.map(score, RUNS))
    lines = ["English only — official F1 ×100 (15 classes incl. None), mean ± std over 5 folds", "",
             "| run | None | thr | EN micro | EN macro |", "|---|---|---|---|---|"]
    best = (-1, None)
    for name, out in res.items():
        for (mode, scope), (per, p) in out.items():
            mi, ma = per.mean(0) * 100, per.std(0) * 100
            lines.append(f"| {name} | {mode} | {scope} | {mi[0]:.1f} ± {ma[0]:.1f} | {mi[1]:.1f} ± {ma[1]:.1f} |")
            if mi[0] > best[0]:
                best = (mi[0], (name, mode, scope, p))
    name, mode, scope, p = best[1]
    t, q = with_none(Y[EN]), with_none(p)
    P, Rr, F1, S = precision_recall_fscore_support(t, q, average=None, zero_division=0)
    lines += ["", f"Per-label EN F1 for the best EN micro: {name} ({mode}, {scope} thr)", "",
              "| label | P | R | F1 | gold | pred |", "|---|---|---|---|---|---|"]
    for i, lab in enumerate(LABELS + ["None"]):
        lines.append(f"| {lab} | {P[i]:.2f} | {Rr[i]:.2f} | {F1[i]:.2f} | {S[i]} | {q[:, i].sum()} |")
    (R / "en_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
