"""Step 9: co-occurrence stacker. Features per paragraph = for each model: its 14 probabilities + the mean of the
previous/next paragraphs' probabilities (same article) ; plus position (para_idx, is first paragraph).
One LightGBM per label. Nested CV: outer fold k = test; inner 4-fold CV on the other folds gives honest stacker
scores to fit thresholds ('global' or 'label'); then the stacker is refit on all 4 folds and applied to fold k."""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np

from src.ens import fit_dec

warnings.filterwarnings("ignore")
PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=15, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


def features(probs: list, article_ids, para_idx):
    art = np.asarray(article_ids); idx = np.asarray(para_idx); n = len(art)
    prev = np.array([i - 1 if i > 0 and art[i - 1] == art[i] else -1 for i in range(n)])
    nxt = np.array([i + 1 if i + 1 < n and art[i + 1] == art[i] else -1 for i in range(n)])
    cols = []
    for P in probs:
        nb = np.zeros_like(P); cnt = np.zeros((n, 1))
        for ref in (prev, nxt):
            ok = ref >= 0; nb[ok] += P[ref[ok]]; cnt[ok] += 1
        cols += [P, nb / np.maximum(cnt, 1)]
    return np.hstack(cols + [idx[:, None], (idx == 0)[:, None].astype(float)])


def fit_predict(Xtr, Ytr, Xte):
    out = np.zeros((len(Xte), Ytr.shape[1]), np.float32)
    for j in range(Ytr.shape[1]):
        if Ytr[:, j].sum() < 3:
            out[:, j] = Ytr[:, j].mean(); continue
        out[:, j] = lgb.LGBMClassifier(**PARAMS, random_state=j).fit(Xtr, Ytr[:, j]).predict_proba(Xte)[:, 1]
    return out


def nested(X, Y, F, kind="global"):
    P = np.zeros(Y.shape, bool); S_all = np.zeros(Y.shape, np.float32)
    for k in range(5):
        tr = np.flatnonzero(F != k); te = np.flatnonzero(F == k)
        inner = np.zeros((len(tr), Y.shape[1]), np.float32)
        for kk in set(F[tr]):
            a, b = F[tr] != kk, F[tr] == kk
            inner[b] = fit_predict(X[tr][a], Y[tr][a], X[tr][b])
        t = fit_dec(inner, Y[tr], kind)
        S_all[te] = fit_predict(X[tr], Y[tr], X[te]); P[te] = S_all[te] >= t
    return P, S_all
