"""Final analyses for the README (no training): paired article bootstrap CIs, per-frame F1, ablation on EN CV,
and the user's model on the held-out 2b EN paragraphs. Writes runs/analysis_final.md.

All CV predictions are nested (weights / thresholds / title factor for fold k chosen on the other 4 folds)."""
import json

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.data import LABELS, ROOT, load_cv, load_extra
from src.en import load_en
from src.ens import fit_dec, weight_grid
from src.post import _micro
from src.step10 import FINE, load_deberta, load_train_cv

out = []
say = lambda s="": (print(s), out.append(s))
cv = load_cv(); em = (cv.lang == "en").values; en, Y, F = load_en(); Ya = np.stack(cv.labels).astype(bool)
mc, _, _ = load_train_cv("phase1b/M_E1_extraEN", em, cv.fold.values, Ya, cv.lang.values)
xl, _, _ = load_train_cv("phase1c/XL_E1", em, cv.fold.values, Ya, cv.lang.values)
db, _, _ = load_deberta("deb_silver_bce", Y, F)
title = en.para_idx.values == 0; arts = en.article_id.values
M = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]


def nested(oofs, dec="global", title_rule=False):
    P = np.zeros(Y.shape, bool)
    for k in range(5):
        tr, te = F != k, F == k; best = (-1, None, None)
        for w in weight_grid(len(oofs), 0.1):
            S = sum(a * o for a, o in zip(w, oofs))
            t = fit_dec(S[tr], Y[tr], "label") if dec == "label" else np.full(14, max(FINE, key=lambda t: _micro(Y[tr], S[tr] >= t)))
            f = _micro(Y[tr], S[tr] >= t)
            if f > best[0]:
                best = (f, w, t)
        _, w, t = best; S = sum(a * o for a, o in zip(w, oofs))
        m = 1.0
        if title_rule:
            m = max(M, key=lambda m: _micro(Y[tr], np.where(title[tr, None], S[tr] >= t * m, S[tr] >= t)))
        P[te] = np.where(title[te, None], S[te] >= t * m, S[te] >= t)
    return P


def micro(P, idx=None):
    return 100 * _micro(Y if idx is None else Y[idx], P if idx is None else P[idx])


ua = np.unique(arts); rows_of = {a: np.flatnonzero(arts == a) for a in ua}
rng = np.random.default_rng(0); BOOT = [np.concatenate([rows_of[a] for a in rng.choice(ua, len(ua))]) for _ in range(2000)]


def boot(PA, PB):
    d = np.array([micro(PB, i) - micro(PA, i) for i in BOOT])
    return micro(PB) - micro(PA), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


S = {
    "mCPT alone": nested([mc]),
    "XLM-R large alone": nested([xl]),
    "silver DeBERTa alone": nested([db]),
    "silver DeBERTa, per-label thresholds": nested([db], "label"),
    "mCPT + XLM-R large": nested([mc, xl]),
    "mCPT + XLM-R large + silver DeBERTa": nested([mc, xl, db]),
    "mCPT + XLM-R large + silver DeBERTa + title rule": nested([mc, xl, db], title_rule=True),
}
per_fold = lambda P: [round(micro(P, np.flatnonzero(F == k)), 1) for k in range(5)]
say("# Final analyses (EN, 5-fold CV, nested; 86 articles / 907 paragraphs)\n")
say("## A. Ablation on CV\n")
say("| system | EN micro | per fold | EN macro |")
say("|---|---|---|---|")
from src.score import official_scores
for n, P in S.items():
    say(f"| {n} | {micro(P):.2f} | {per_fold(P)} | {100 * official_scores(Y, P)[1]:.1f} |")
say("")
say("## B. Paired bootstrap over articles (2000 resamples): gain of B over A, 95% CI, share of resamples with a gain\n")
say("| A | B | gain | 95% CI | P(gain > 0) |")
say("|---|---|---|---|---|")
for a, b in [("silver DeBERTa, per-label thresholds", "silver DeBERTa alone"),
             ("mCPT alone", "mCPT + XLM-R large"),
             ("mCPT + XLM-R large", "mCPT + XLM-R large + silver DeBERTa"),
             ("mCPT alone", "mCPT + XLM-R large + silver DeBERTa"),
             ("mCPT + XLM-R large + silver DeBERTa", "mCPT + XLM-R large + silver DeBERTa + title rule")]:
    g, lo, hi, p = boot(S[a], S[b])
    say(f"| {a} | {b.replace('mCPT + XLM-R large + silver DeBERTa', 'ensemble')} | {g:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {p:.2f} |")
say("")
say("## C. Per-frame F1 of the final CV system, without / with the title rule\n")
P0, P1 = S["mCPT + XLM-R large + silver DeBERTa"], S["mCPT + XLM-R large + silver DeBERTa + title rule"]
f1 = lambda y, p: 2 * (y & p).sum() / max(1, y.sum() + p.sum())
say("| frame | gold n | F1 | F1 + title rule |")
say("|---|---|---|---|")
for j, l in enumerate(LABELS + ["None"]):
    y = Y[:, j] if j < 14 else ~Y.any(1); a = P0[:, j] if j < 14 else ~P0.any(1); b = P1[:, j] if j < 14 else ~P1.any(1)
    say(f"| {l} | {y.sum()} | {f1(y, a):.2f} | {f1(y, b):.2f} |")
say("")
for n, P in [("without", P0), ("with", P1)]:
    t15 = np.c_[Y, ~Y.any(1)]; p15 = np.c_[P, ~P.any(1)]; tp = (t15 & p15).sum(); fp = (~t15 & p15).sum(); fn = (t15 & ~p15).sum()
    say(f"- {n} title rule: precision {tp / (tp + fp):.3f}, recall {tp / (tp + fn):.3f}, titles predicted None "
        f"{(~P[title].any(1)).mean():.0%} (gold {(~Y[title].any(1)).mean():.0%}), title micro {micro(P, np.flatnonzero(title)):.1f}, "
        f"body micro {micro(P, np.flatnonzero(~title)):.1f}")
say("")

# D. user's model on the held-out 2b EN paragraphs
B = ROOT / "ckpt/EN_best_model"; meta = json.load(open(B / "meta.json")); order = [meta["labels"].index(l) for l in LABELS]
ex = load_extra(("en",)); v = ex[ex.article_id.str.startswith("D2B_")].reset_index(drop=True)
Yv = np.stack(v.labels).astype(bool); tv = (v.title.values == v.text.values)
tok = AutoTokenizer.from_pretrained(B / "encoder"); enc = AutoModel.from_pretrained(B / "encoder").cuda().eval()
head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 14)).cuda().eval()
head.load_state_dict(torch.load(B / "head.pt", weights_only=True))
with torch.no_grad():
    pr = []
    for b in range(0, len(v), 32):
        t_ = tok(v.text.tolist()[b:b + 32], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        h = enc(**t_).last_hidden_state; mm = t_["attention_mask"].unsqueeze(-1).float(); pr.append(torch.sigmoid(head((h * mm).sum(1) / mm.sum(1))).cpu().numpy())
Pv = np.concatenate(pr)[:, order]; A0 = Pv >= 0.3; A1 = np.where(tv[:, None], Pv >= 0.9, Pv >= 0.3)
say("## D. EN_best_model on the 224 held-out 2b EN paragraphs (never trained on; used by the notebook for selection)\n")
say(f"- micro {100 * _micro(Yv, A0):.2f} -> {100 * _micro(Yv, A1):.2f} with the title rule (3x threshold); titles {tv.sum()}, gold None among titles {(~Yv[tv].any(1)).mean():.0%}")
say("")
say("| frame | gold n | F1 | F1 + title rule |")
say("|---|---|---|---|")
for j, l in enumerate(LABELS + ["None"]):
    y = Yv[:, j] if j < 14 else ~Yv.any(1); a = A0[:, j] if j < 14 else ~A0.any(1); b = A1[:, j] if j < 14 else ~A1.any(1)
    say(f"| {l} | {y.sum()} | {f1(y, a):.2f} | {f1(y, b):.2f} |")
(ROOT / "runs/analysis_final.md").write_text("\n".join(out) + "\n"); print("\nwrote runs/analysis_final.md")
