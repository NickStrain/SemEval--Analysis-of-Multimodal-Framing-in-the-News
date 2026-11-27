"""Turning 15-column scores into label sets, threshold tuning, and cross-fitted selection.

Scores S: (N, 15) = 14 frame scores + a None score (probabilities or cosines).

None handling modes:
  implicit : predict frame l iff S[:, l] >= T[l]; None iff nothing clears (15th column ignored).
  explicit : None iff S[:, 14] >= T[14]  (-> empty set, overrides frames);
             otherwise frames clearing their threshold, and if none clears, the single frame with
             the largest margin S - T (the model said "not None", so emit something).

Cross-fitting: for test fold k, the checkpoint (epoch / C) and thresholds are chosen using only
the out-of-fold scores of the other 4 folds. Fold k's labels are never used for any choice.
(Those other-fold scores come from models that were trained on fold k's data, so this is honest
for thresholds/epochs but not a fully nested CV.)
"""
from __future__ import annotations

import numpy as np

N_LAB = 14


def with_none(y14: np.ndarray) -> np.ndarray:
    y14 = y14.astype(bool)
    return np.concatenate([y14, ~y14.any(1, keepdims=True)], 1)


def micro_f1_fast(t15: np.ndarray, p15: np.ndarray) -> float:
    tp = np.count_nonzero(t15 & p15)
    fp = np.count_nonzero(~t15 & p15)
    fn = np.count_nonzero(t15 & ~p15)
    return 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def decode(S: np.ndarray, T: np.ndarray, mode: str) -> np.ndarray:
    """-> (N,14) bool."""
    P = S[:, :N_LAB] >= T[:N_LAB]
    if mode == "implicit":
        return P
    none = S[:, N_LAB] >= T[N_LAB]
    fb = ~none & ~P.any(1)
    if fb.any():
        P[np.flatnonzero(fb), np.argmax(S[fb, :N_LAB] - T[:N_LAB], 1)] = True
    P[none] = False
    return P


def _cands(col: np.ndarray) -> np.ndarray:
    return np.unique(np.quantile(col, np.linspace(0.01, 0.99, 61)))


def tune(S: np.ndarray, Y14: np.ndarray, mode: str, passes: int = 3) -> np.ndarray:
    """Per-label thresholds maximising official micro-F1 (15 classes) by coordinate ascent."""
    t15 = with_none(Y14)
    n_cols = N_LAB + (mode == "explicit")
    cands = [_cands(S[:, j]) for j in range(n_cols)]
    T = np.full(N_LAB + 1, np.inf)
    # init: each column at its own best binary F1
    for j in range(n_cols):
        y = t15[:, j]
        best = max(cands[j], key=lambda c: 2 * np.count_nonzero(y & (S[:, j] >= c))
                   / max(1, y.sum() + np.count_nonzero(S[:, j] >= c)))
        T[j] = best
    cur = micro_f1_fast(t15, with_none(decode(S, T, mode)))
    for _ in range(passes):
        improved = False
        for j in range(n_cols):
            for c in cands[j]:
                old = T[j]
                T[j] = c
                f = micro_f1_fast(t15, with_none(decode(S, T, mode)))
                if f > cur + 1e-12:
                    cur, improved = f, True
                else:
                    T[j] = old
        if not improved:
            break
    return T


def crossfit(scores: np.ndarray, Y14: np.ndarray, folds: np.ndarray, langs: np.ndarray,
             mode: str, per_lang: bool) -> tuple[np.ndarray, list[int]]:
    """scores: (C, N, 15) OOF scores for C checkpoints. Returns (N,14) bool predictions and the
    checkpoint chosen for each fold."""
    pred = np.zeros(Y14.shape, bool)
    chosen = []
    for k in np.unique(folds):
        sel, test = folds != k, folds == k
        best = (-1, None, None)
        for c in range(scores.shape[0]):
            S = scores[c, sel]
            if np.isnan(S[:, : N_LAB + (mode == "explicit")]).any():
                continue
            T = tune(S, Y14[sel], mode)
            f = micro_f1_fast(with_none(Y14[sel]), with_none(decode(S, T, mode)))
            if f > best[0]:
                best = (f, c, T)
        _, c, T = best
        chosen.append(int(c))
        if not per_lang:
            pred[test] = decode(scores[c, test], T, mode)
        else:
            for lg in np.unique(langs[test]):
                s_l, t_l = sel & (langs == lg), test & (langs == lg)
                T_l = tune(scores[c, s_l], Y14[s_l], mode)
                pred[t_l] = decode(scores[c, t_l], T_l, mode)
    return pred, chosen
