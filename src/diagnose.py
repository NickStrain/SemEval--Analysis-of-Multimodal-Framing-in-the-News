"""Diagnosis of the best EN system (mCPT + XLM-R large + silver DeBERTa, global threshold): where micro-F1 is lost.

All numbers on EN CV out-of-fold scores. 'nested' = weights/threshold chosen without the test fold (honest);
'oracle' rows cheat on purpose to measure how much each sub-problem would be worth if solved.
Writes runs/diagnosis.md.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import LABELS, ROOT, load_cv
from src.en import load_en
from src.ens import weight_grid
from src.post import _micro
from src.step10 import FINE, load_deberta, load_train_cv, nested_gen

out = []
say = lambda s="": (print(s), out.append(s))


def per_fold_micro(Y, P, F):
    return np.array([_micro(Y[F == k], P[F == k]) for k in range(5)])


cv = load_cv(); Y_all = np.stack(cv.labels).astype(bool); en_mask = (cv.lang == "en").values
en, Y, F = load_en()
mc, _, _ = load_train_cv("phase1b/M_E1_extraEN", en_mask, cv.fold.values, Y_all, cv.lang.values)
xl, _, _ = load_train_cv("phase1c/XL_E1", en_mask, cv.fold.values, Y_all, cv.lang.values)
db, _, _ = load_deberta("deb_silver_bce", Y, F)
oofs = [mc, xl, db]

# ensemble score per fold with weights chosen on the other folds (same as nested), keep the scores
S = np.zeros(Y.shape, np.float32); T = np.zeros(5)
for k in range(5):
    tr, te = F != k, F == k; best = (-1, None, None)
    for w in weight_grid(3, 0.1):
        s = sum(wi * o for wi, o in zip(w, oofs))
        t = max(FINE, key=lambda t: _micro(Y[tr], s[tr] >= t)); f = _micro(Y[tr], s[tr] >= t)
        if f > best[0]:
            best = (f, w, t)
    S[te] = sum(wi * o[te] for wi, o in zip(best[1], oofs)); T[k] = best[2]
P = np.zeros(Y.shape, bool)
for k in range(5):
    P[F == k] = S[F == k] >= T[k]
base = per_fold_micro(Y, P, F)
say("# Diagnosis of the best EN system (mCPT + XLM-R large + silver DeBERTa)\n")
say(f"Nested micro {100 * base.mean():.2f} ± {100 * base.std():.2f}; thresholds chosen per fold: {T.round(2).tolist()}\n")

# ---------- 1. data ----------
n = len(Y); card = Y.sum(1)
say("## 1. Data")
say(f"- EN training paragraphs: {n} in {en.article_id.nunique()} articles (tiny for a 15-way multi-label task)")
say(f"- gold labels per paragraph: mean {card.mean():.2f}; distribution 0:{(card == 0).mean():.0%} 1:{(card == 1).mean():.0%} "
    f"2:{(card == 2).mean():.0%} 3+:{(card >= 3).mean():.0%}")
say(f"- predicted labels per paragraph: mean {P.sum(1).mean():.2f}; predicted None {(P.sum(1) == 0).mean():.0%} vs gold None {(card == 0).mean():.0%}")
say("")

# ---------- 2. ranking vs decision ----------
say("## 2. Is the model ranking frames badly, or deciding badly?")
say("| label | gold n | AUC | AP | F1 (nested) |")
say("|---|---|---|---|---|")
f1s = []
for j, l in enumerate(LABELS):
    y = Y[:, j]; tp = (y & P[:, j]).sum(); f1 = 2 * tp / max(1, y.sum() + P[:, j].sum()); f1s.append(f1)
    auc = roc_auc_score(y, S[:, j]) if 0 < y.sum() < n else float("nan")
    ap = average_precision_score(y, S[:, j]) if y.sum() else float("nan")
    say(f"| {l} | {y.sum()} | {auc:.2f} | {ap:.2f} | {f1:.2f} |")
say("")

# ---------- 3. oracles ----------
say("## 3. Upper bounds: what each sub-problem is worth if solved (cheating on purpose)")
rows = []
# a. oracle threshold per fold
Po = np.zeros_like(P)
for k in range(5):
    m = F == k; Po[m] = S[m] >= max(FINE, key=lambda t: _micro(Y[m], S[m] >= t))
rows.append(("threshold tuned on the test fold itself", per_fold_micro(Y, Po, F)))
# b. oracle per-label thresholds on the test fold
Pl = np.zeros_like(P)
for k in range(5):
    m = F == k
    for j in range(14):
        Pl[m, j] = S[m, j] >= max(FINE, key=lambda t: _micro(Y[m][:, [j]], S[m][:, [j]] >= t) if False else
                                  (2 * ((S[m, j] >= t) & Y[m, j]).sum() / max(1, (S[m, j] >= t).sum() + Y[m, j].sum())))
rows.append(("per-label thresholds tuned on test fold", per_fold_micro(Y, Pl, F)))
# c. oracle None: known which paragraphs have no frame
Pn = P.copy(); Pn[card == 0] = False
gold_framed_empty = (card > 0) & (P.sum(1) == 0)
top = np.argmax(S, 1); Pn[gold_framed_empty, top[gold_framed_empty]] = True
rows.append(("None decision perfect (frames: best score if empty)", per_fold_micro(Y, Pn, F)))
# d. oracle count: know how many labels each paragraph has, take top-k by score
order = np.argsort(-S, 1); Pk = np.zeros_like(P)
for i in range(n):
    Pk[i, order[i, :card[i]]] = True
rows.append(("number of labels known, top-k by score", per_fold_micro(Y, Pk, F)))
# e. top-1 only with perfect None
P1 = np.zeros_like(P); P1[np.arange(n), top] = True; P1[card == 0] = False
rows.append(("perfect None + top-1 frame only", per_fold_micro(Y, P1, F)))
# f. article gold frames known: restrict predictions to frames used in the article
arts = en.article_id.values
G = {a: Y[arts == a].any(0) for a in np.unique(arts)}
mask = np.stack([G[a] for a in arts])
Pa = np.zeros_like(P)
for k in range(5):
    m = F == k; Sm = np.where(mask[m], S[m], 0)
    Pa[m] = Sm >= max(FINE, key=lambda t: _micro(Y[m], Sm >= t))
rows.append(("article's gold frame set known (+ test-fold thr)", per_fold_micro(Y, Pa, F)))
# g. trivial: every paragraph gets all its article's gold frames
rows.append(("predict ALL article gold frames on every paragraph", per_fold_micro(Y, mask.copy(), F)))
say("| scenario | micro |")
say("|---|---|")
say(f"| **current system (honest)** | **{100 * base.mean():.1f}** |")
for name, r in rows:
    say(f"| {name} | {100 * r.mean():.1f} |")
say("")

# ---------- 4. where the errors are ----------
say("## 4. Error budget (nested predictions)")
t15 = np.c_[Y, card == 0]; p15 = np.c_[P, P.sum(1) == 0]
tp = (t15 & p15).sum(0); fp = (~t15 & p15).sum(0); fn = (t15 & ~p15).sum(0)
TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
say(f"Totals: TP {TP}, FP {FP}, FN {FN} -> precision {TP / (TP + FP):.2f}, recall {TP / (TP + FN):.2f}")
say("| label | TP | FP | FN | micro gain if fixed |")
say("|---|---|---|---|---|")
names = LABELS + ["None"]
gain = [(2 * TP / (2 * TP + FP - fp[j] + FN - fn[j] + 0) if True else 0) for j in range(15)]
for j in np.argsort([-(g) for g in gain]):
    g = 2 * (TP + fn[j]) / (2 * (TP + fn[j]) + FP - fp[j] + FN - fn[j]) - 2 * TP / (2 * TP + FP + FN)
    say(f"| {names[j]} | {tp[j]} | {fp[j]} | {fn[j]} | +{100 * g:.1f} |")
say("")

# ---------- 5. confusions: which frames are mistaken for which ----------
say("## 5. Most common confusions (predicted X where gold has Y but not X)")
conf = {}
for i in range(n):
    for a in np.flatnonzero(P[i] & ~Y[i]):
        for b in np.flatnonzero(Y[i] & ~P[i]):
            conf[(a, b)] = conf.get((a, b), 0) + 1
for (a, b), c in sorted(conf.items(), key=lambda x: -x[1])[:10]:
    say(f"- predicted {LABELS[a]} instead of {LABELS[b]}: {c}")
say("")

# ---------- 6. position / title / length ----------
say("## 6. By paragraph type")
is_title = en.para_idx.values == 0
L = en.text.str.split().str.len().values
for name, m in [("title (para 0)", is_title), ("body", ~is_title), ("<25 words", L < 25), ("25-80 words", (L >= 25) & (L < 80)),
                (">=80 words", L >= 80)]:
    say(f"- {name}: {m.sum()} paras, gold None {(card[m] == 0).mean():.0%}, pred None {(P[m].sum(1) == 0).mean():.0%}, "
        f"micro {100 * _micro(Y[m], P[m]):.1f}")
say("")

# ---------- 7. label consistency between neighbours ----------
same_art_next = arts[1:] == arts[:-1]
jacc = [(Y[i] & Y[i + 1]).sum() / max(1, (Y[i] | Y[i + 1]).sum()) for i in np.flatnonzero(same_art_next)]
say("## 7. Context")
say(f"- adjacent paragraphs: mean label Jaccard {np.mean(jacc):.2f}; a paragraph's frames appear elsewhere in its article "
    f"{(Y & mask).sum() / max(1, Y.sum()):.0%} of the time (by definition 100%); article has {np.mean([G[a].sum() for a in G]):.1f} frames on average")
(ROOT / "runs" / "diagnosis.md").write_text("\n".join(out) + "\n")
print("\nwrote runs/diagnosis.md")
