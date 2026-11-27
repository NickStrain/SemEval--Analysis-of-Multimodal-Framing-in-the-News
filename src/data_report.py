"""Phase 0.6: label statistics. Computed on train+practice+gate only (final stays locked)."""
import json
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from src.data import LABELS, LANGS, SPLITS_FILE, load_all

df_all = load_all()
_sp = json.loads(SPLITS_FILE.read_text())
sp = {f"fold{k}": ids for k, ids in enumerate(_sp["folds"])} | {"final": _sp["final"]}

print("== Split sizes (articles / paragraphs) ==")
for name, ids in sp.items():
    sub = df_all[df_all.article_id.isin(set(ids))]
    by = sub.groupby("lang").article_id.nunique().reindex(LANGS, fill_value=0)
    print(f"{name:<9} {len(ids):>4} art {len(sub):>5} para   " + "  ".join(f"{l}:{by[l]}" for l in LANGS))

df = df_all[~df_all.article_id.isin(set(sp["final"]))]
Y = np.stack(df.labels)
print(f"\n== Stats below: train+practice+gate, {df.article_id.nunique()} articles, {len(df)} paragraphs (final excluded) ==")

freq = pd.DataFrame({lg: Y[(df.lang == lg).values].sum(0) for lg in LANGS}, index=LABELS)
freq["ALL"] = freq.sum(axis=1)
npara = {lg: int((df.lang == lg).sum()) for lg in LANGS} | {"ALL": len(df)}
print("\n== Label frequency per language (count, % of paragraphs) ==")
print(freq.apply(lambda c: [f"{v} ({100 * v / npara[c.name]:.0f}%)" for v in c]).to_string())

k = Y.sum(1)
print("\n== Labels per paragraph ==")
for lg in LANGS + ["ALL"]:
    m = (df.lang == lg).values if lg != "ALL" else np.ones(len(df), bool)
    print(f"{lg:<4} mean {k[m].mean():.2f}   mean|labelled {k[m][k[m] > 0].mean():.2f}   "
          f"NONE {100*(k[m] == 0).mean():.1f}%   max {k[m].max()}")

pairs = Counter()
for row in Y:
    idx = np.flatnonzero(row)
    pairs.update(combinations(idx, 2))
print("\n== Top-10 label pairs (co-occurring in a paragraph) ==")
for (a, b), c in pairs.most_common(10):
    print(f"{c:>5}  {LABELS[a]} + {LABELS[b]}")
