"""Step 2 post-processing on (N,14) probabilities:
   per-label thresholds t_l (0.05..0.95, step 0.05), None rule (None if max prob < m, m in 0.1..0.6 or off),
   cap k labels (k in 1,2,3 or none; keeps the k highest among those above threshold).
Fit = grid over (m, k) x coordinate ascent on official micro-F1 over the thresholds.
cv_apply = cross-fitted (fit on 4 folds, apply to the 5th) -> honest OOF predictions."""
from __future__ import annotations

import numpy as np

GRID_T = np.round(np.arange(0.05, 0.951, 0.05), 2)
GRID_M = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
GRID_K = [None, 1, 2, 3]


def apply(S, prm):
    t, m, k = np.asarray(prm["t"]), prm["m"], prm["k"]
    P = S >= t
    if k:
        order = np.argsort(-S, 1)
        rank = np.empty_like(order); np.put_along_axis(rank, order, np.arange(S.shape[1])[None].repeat(len(S), 0), 1)
        P &= rank < k
    if m is not None:
        P[S.max(1) < m] = False
    return P


def _micro(Y, P):
    t = np.c_[Y, ~Y.any(1)]; p = np.c_[P, ~P.any(1)]
    tp = (t & p).sum(); fp = (~t & p).sum(); fn = (t & ~p).sum()
    return 2 * tp / max(1, 2 * tp + fp + fn)


def fit(S, Y, passes=2):
    best = (-1, None)
    for m in GRID_M:
        for k in GRID_K:
            prm = {"t": [0.5] * S.shape[1], "m": m, "k": k}
            cur = _micro(Y, apply(S, prm))
            for _ in range(passes):
                for j in range(S.shape[1]):
                    for c in GRID_T:
                        old = prm["t"][j]; prm["t"][j] = float(c)
                        f = _micro(Y, apply(S, prm))
                        if f > cur + 1e-12:
                            cur = f
                        else:
                            prm["t"][j] = old
            if cur > best[0]:
                best = (cur, {"t": list(prm["t"]), "m": m, "k": k})
    return best[1]


def cv_apply(S, Y, folds):
    P = np.zeros(Y.shape, bool); prms = []
    for k in range(5):
        prm = fit(S[folds != k], Y[folds != k]); prms.append(prm)
        P[folds == k] = apply(S[folds == k], prm)
    return P, prms
