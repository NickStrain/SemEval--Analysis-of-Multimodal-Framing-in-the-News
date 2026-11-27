Official F1 ×100, mean ± std over 5 folds (None: implicit, global thresholds, cross-fitted).
Δ = mean change vs E0 (folds better /5); ✅ = passes the rule (≥ +0.5 and ≥ 4/5 folds).

### mCPT (ctx, lr 5e-5)

| run | micro | macro | **EN** | BG | PT | RU | Δ EN | Δ micro | Δ macro |
|---|---|---|---|---|---|---|---|---|---|
| E0 | 51.8 ± 0.9 | 42.4 ± 1.3 | 35.6 ± 3.8 | 44.6 ± 4.1 | 62.0 ± 3.6 | 60.1 ± 3.2 | – | – | – |
| E1 +batch-1 EN | 52.4 ± 1.2 | 42.8 ± 2.2 | 36.8 ± 2.4 | 43.4 ± 4.2 | 63.7 ± 3.3 | 58.7 ± 4.0 | +1.2 (4/5) ✅ | +0.6 (5/5) ✅ | +0.4 (3/5) |
| E3 +batch-1 EN, EN x2 | 51.9 ± 1.2 | 42.1 ± 1.7 | 36.2 ± 2.5 | 44.0 ± 4.9 | 62.4 ± 3.4 | 58.3 ± 3.1 | +0.6 (3/5) | +0.1 (3/5) | -0.4 (1/5) |

English specialist (EN rows only, thresholds tuned on EN only):

| run | EN micro | EN macro | Δ EN micro vs E0 | Δ vs E3 |
|---|---|---|---|---|
| E0 | 33.9 ± 4.7 | 29.0 ± 4.3 | – | – |
| E3 +batch-1 EN, EN x2 | 34.3 ± 2.0 | 29.6 ± 0.9 | +0.5 (3/5) | – |
| E4 = E3 + EN-only stage 2 | 33.8 ± 2.1 | 27.7 ± 1.5 | -0.1 (2/5) | -0.6 (2/5) |

### XLM-R base (lr 2e-5)

| run | micro | macro | **EN** | BG | PT | RU | Δ EN | Δ micro | Δ macro |
|---|---|---|---|---|---|---|---|---|---|
| E0 | 52.8 ± 0.7 | 43.9 ± 2.3 | 35.5 ± 3.7 | 43.8 ± 6.6 | 64.3 ± 2.3 | 60.5 ± 2.7 | – | – | – |

English specialist (EN rows only, thresholds tuned on EN only):

| run | EN micro | EN macro | Δ EN micro vs E0 | Δ vs E3 |
|---|---|---|---|---|
| E0 | 35.3 ± 5.7 | 29.9 ± 6.7 | – | – |
