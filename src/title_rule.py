"""Test: stricter threshold for title paragraphs (para 0) -> more None. Honest checks only.
(1) our EN CV, best ensemble mCPT+XLM-R L+silver DeBERTa: title multiplier m chosen on 4 folds, applied to the 5th.
(2) user's untouched model on the 224 held-out 2b EN paragraphs (it was never trained on them)."""
import json, numpy as np, torch, torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from src.data import ROOT, load_cv, load_extra
from src.en import load_en, per_fold
from src.ens import weight_grid
from src.post import _micro
from src.step10 import FINE, load_deberta, load_train_cv
cv = load_cv(); em = (cv.lang == "en").values; en, Y, F = load_en(); Ya = np.stack(cv.labels).astype(bool)
mc, _, _ = load_train_cv("phase1b/M_E1_extraEN", em, cv.fold.values, Ya, cv.lang.values)
xl, _, _ = load_train_cv("phase1c/XL_E1", em, cv.fold.values, Ya, cv.lang.values)
db, _, _ = load_deberta("deb_silver_bce", Y, F)
title = en.para_idx.values == 0; M = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]
def dec(S, t, m, tit): return np.where(tit[:, None], S >= t * m, S >= t)
P0 = np.zeros(Y.shape, bool); P1 = np.zeros(Y.shape, bool); ch = []
for k in range(5):
    tr, te = F != k, F == k
    w = max(weight_grid(3, 0.1), key=lambda w: max(_micro(Y[tr], sum(a * o for a, o in zip(w, [mc, xl, db]))[tr] >= t) for t in FINE[::2]))
    S = sum(a * o for a, o in zip(w, [mc, xl, db])); t = max(FINE, key=lambda t: _micro(Y[tr], S[tr] >= t))
    m = max(M, key=lambda m: _micro(Y[tr], dec(S[tr], t, m, title[tr]))); ch.append(m)
    P0[te] = S[te] >= t; P1[te] = dec(S[te], t, m, title[te])
for n, P in [("no title rule", P0), ("title rule (nested)", P1)]:
    mi, _ = per_fold(Y, P, F); print(f"CV EN  {n:22s} {100 * mi.mean():.2f} ± {100 * mi.std():.2f}  folds {np.round(100 * mi, 1)}")
print("chosen title multipliers per fold:", ch, "| title None: gold", f"{(~Y[title].any(1)).mean():.0%}", "pred", f"{(~P0[title].any(1)).mean():.0%} -> {(~P1[title].any(1)).mean():.0%}")
# (2) user's model on held-out 2b EN
B = ROOT / "ckpt/EN_best_model"; meta = json.load(open(B / "meta.json")); order = [meta["labels"].index(l) for l in __import__("src.data", fromlist=["LABELS"]).LABELS]
ex = load_extra(("en",)); v = ex[ex.article_id.str.startswith("D2B_")].reset_index(drop=True)
Yv = np.stack(v.labels).astype(bool); tv = (v.title.values == v.text.values)
tok = AutoTokenizer.from_pretrained(B / "encoder"); enc = AutoModel.from_pretrained(B / "encoder").cuda().eval()
head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 14)).cuda().eval(); head.load_state_dict(torch.load(B / "head.pt", weights_only=True))
with torch.no_grad():
    out = []
    for b in range(0, len(v), 32):
        t_ = tok(v.text.tolist()[b:b + 32], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        h = enc(**t_).last_hidden_state; mm = t_["attention_mask"].unsqueeze(-1).float(); out.append(torch.sigmoid(head((h * mm).sum(1) / mm.sum(1))).cpu().numpy())
Pv = np.concatenate(out)[:, order]
print(f"\n2b EN held-out: {tv.sum()} title paragraphs, gold None among titles {(~Yv[tv].any(1)).mean():.0%}")
for m in M:
    print(f"  user model thr 0.3, title multiplier {m}: micro {100 * _micro(Yv, dec(Pv, 0.3, m, tv)):.2f}")
