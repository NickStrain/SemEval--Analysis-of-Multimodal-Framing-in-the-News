"""Nested-CV decoding + weighted ensembling on EN OOF probabilities (N,14).

dec kinds:  'label'  per-label thresholds (0.05..0.95) by coordinate ascent on micro
            'global' one threshold for all labels
For each test fold k: choose ensemble weights AND thresholds on the other 4 folds only, apply to fold k.
"""
from __future__ import annotations

import itertools

import numpy as np

from src.post import GRID_T, _micro


def fit_dec(S, Y, kind):
    if kind == "global":
        best = max(GRID_T, key=lambda t: _micro(Y, S >= t)); return np.full(S.shape[1], best)
    t = np.full(S.shape[1], 0.5); cur = _micro(Y, S >= t)
    for _ in range(2):
        for j in range(S.shape[1]):
            for c in GRID_T:
                old = t[j]; t[j] = c; f = _micro(Y, S >= t)
                if f > cur + 1e-12:
                    cur = f
                else:
                    t[j] = old
    return t


def weight_grid(n, step=0.05):
    if n == 1:
        return [np.array([1.0])]
    g = np.round(np.arange(0, 1 + 1e-9, step), 2)
    return [np.array(w + (1 - sum(w),)) for w in itertools.product(g, repeat=n - 1) if sum(w) <= 1 + 1e-9]


def nested(oofs, Y, F, kind="label", step=0.05):
    """oofs: list of (N,14). Returns predictions (N,14) bool and chosen weights per fold."""
    P = np.zeros(Y.shape, bool); chosen = []
    for k in range(5):
        tr, te = F != k, F == k; best = (-1, None, None)
        for w in weight_grid(len(oofs), step):
            S = sum(wi * o for wi, o in zip(w, oofs))
            t = fit_dec(S[tr], Y[tr], kind); f = _micro(Y[tr], S[tr] >= t)
            if f > best[0]:
                best = (f, w, t)
        _, w, t = best; chosen.append(w.round(2).tolist())
        P[te] = sum(wi * o[te] for wi, o in zip(w, oofs)) >= t
    return P, chosen
