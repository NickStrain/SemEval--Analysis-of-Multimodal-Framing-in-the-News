"""Continue training a COPY of the user's EN_best_model (ckpt/EN_best_model is only read, never written).

Train : EN gold it already saw (CV EN + batch-1 EN, replay against forgetting)
        + NEW data: 3454 NLLB English translations + dev-release 2b gold BG/PT (originals).
Val   : dev-release 2b gold EN (224 paragraphs) - never trained on by the user's model or here.
Each epoch: val micro-F1 (official 15-class incl. None) at the best global threshold; when it improves, the whole
model is saved to its OWN folder  ckpt/EN_best_model_ft/<run>/best_e<N>/  (encoder, head.pt, meta.json, dev
predictions + checked zip). Epoch 0 = the untouched model (reference, not saved again).
Progress: runs/ext_ft_<run>.log (one JSON line per step) -> bash runs/track_ft.sh
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.data import LABELS, ROOT, _load_2b_gold, load_cv, load_extra
from src.final_model import load_dev
from src.post import _micro

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="ft1"); ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--head-lr", type=float, default=1e-4)
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--max-len", type=int, default=512)
a = ap.parse_args()
SRC = ROOT / "ckpt/EN_best_model"; OUT = ROOT / "ckpt/EN_best_model_ft" / a.run; OUT.mkdir(parents=True, exist_ok=True)
LOG = open(ROOT / f"runs/ext_ft_{a.run}.log", "a")
def log(**k):
    k["time"] = time.strftime("%H:%M:%S"); print(json.dumps(k), flush=True); LOG.write(json.dumps(k) + "\n"); LOG.flush()

meta = json.load(open(SRC / "meta.json")); order = [LABELS.index(l) for l in meta["labels"]]  # our order -> theirs
torch.manual_seed(0); np.random.seed(0)

# ---------------- data ----------------
cv = load_cv(); ex = load_extra(("en",))
is2b = ex.article_id.str.startswith("D2B_")
val = ex[is2b].reset_index(drop=True)                                   # 224 EN 2b gold, held out
tr_parts = [cv[cv.lang == "en"], ex[~is2b], pd.read_parquet(ROOT / "data/trans/trans_en.parquet"),
            _load_2b_gold("bg"), _load_2b_gold("pt")]
train = pd.concat([p[["text", "labels"]] for p in tr_parts], ignore_index=True)
Ytr = np.stack([np.asarray(y)[order] for y in train.labels]).astype(np.float32)
Yval = np.stack([np.asarray(y) for y in val.labels]).astype(bool)      # our order, for scoring
log(event="data", epochs=a.epochs, train=len(train), val=len(val), new_rows=int(sum(len(p) for p in tr_parts[2:])))

# ---------------- model: a copy of the user's encoder + head ----------------
tok = AutoTokenizer.from_pretrained(SRC / "encoder")
enc = AutoModel.from_pretrained(SRC / "encoder").cuda()
enc.gradient_checkpointing_enable()
head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 14)).cuda()
head.load_state_dict(torch.load(SRC / "head.pt", map_location="cuda", weights_only=True))


def logits(texts):
    t = tok(texts, padding=True, truncation=True, max_length=a.max_len, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = enc(**t).last_hidden_state
    m = t["attention_mask"].unsqueeze(-1).float(); z = (h.float() * m).sum(1) / m.sum(1)
    return head(z)


@torch.no_grad()
def probs(texts):
    enc.eval(); head.eval(); out = []
    for b in range(0, len(texts), 32):
        out.append(torch.sigmoid(logits(texts[b:b + 32])).float().cpu().numpy())
    enc.train(); head.train()
    return np.concatenate(out)[:, np.argsort(order)]                      # back to our label order


GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
def evaluate(ep):
    p = probs(val.text.tolist())
    t = max(GRID, key=lambda t: _micro(Yval, p >= t))
    return dict(epoch=ep, val_micro=float(round(100 * _micro(Yval, p >= t), 2)), thr=float(t),
                val_micro_at_0_3=float(round(100 * _micro(Yval, p >= 0.3), 2)))


dev = load_dev("en")
def save_best(r):
    d = OUT / f"best_e{r['epoch']}"; d.mkdir(exist_ok=True)
    enc.save_pretrained(d / "encoder"); tok.save_pretrained(d / "encoder"); torch.save(head.state_dict(), d / "head.pt")
    m = dict(meta, name=f"EN_best_model_ft/{a.run}/best_e{r['epoch']}", threshold=r["thr"], parent="ckpt/EN_best_model",
             finetune=vars(a), val_2b_en=r)
    json.dump(m, open(d / "meta.json", "w"), indent=1)
    P = probs(dev.text.tolist()); np.save(d / "dev_scores.npy", P)
    sub = ROOT / "runs/submissions" / f"ft_{a.run}_e{r['epoch']}_EN_dev.txt"
    with open(sub, "w", encoding="utf-8", newline="\n") as f:
        for row, p in zip(dev.itertuples(), P >= r["thr"]):
            f.write("\t".join([row.doc, row.start, row.end] + [LABELS[i] for i in np.flatnonzero(p)]) + "\n")
    chk = subprocess.run([sys.executable, "-m", "src.make_zip", str(sub), "en"], capture_output=True, text=True, cwd=ROOT)
    shutil.copy(sub.with_suffix(".zip"), d / sub.with_suffix(".zip").name)
    log(event="saved_best", folder=str(d.relative_to(ROOT)), zip=str(sub.with_suffix(".zip").relative_to(ROOT)),
        format_ok="format is correct" in (chk.stdout + chk.stderr))


r0 = evaluate(0); r0["note"] = "untouched user model (reference)"; log(event="eval", **r0)
best = r0["val_micro"]
opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": a.lr}, {"params": head.parameters(), "lr": a.head_lr}], weight_decay=0.01)
steps = math.ceil(len(train) / (a.bs * a.accum)) * a.epochs
sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
texts = train.text.tolist()
for ep in range(1, a.epochs + 1):
    t0 = time.time(); perm = np.random.permutation(len(train)); tot = 0.0; nb = math.ceil(len(perm) / a.bs)
    for i in range(nb):
        idx = perm[i * a.bs:(i + 1) * a.bs]
        loss = nn.functional.binary_cross_entropy_with_logits(logits([texts[j] for j in idx]).float(),
                                                              torch.from_numpy(Ytr[idx]).cuda())
        (loss / a.accum).backward(); tot += loss.item()
        if (i + 1) % a.accum == 0 or i == nb - 1:
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        if (i + 1) % 100 == 0:
            log(event="progress", epoch=ep, batch=i + 1, of=nb, loss=round(tot / (i + 1), 4), min=round((time.time() - t0) / 60, 1))
    r = evaluate(ep); r["train_loss"] = round(tot / nb, 4); r["min"] = round((time.time() - t0) / 60, 1)
    r["improved"] = bool(r["val_micro"] > best); log(event="eval", **r)
    if r["improved"]:
        best = r["val_micro"]; save_best(r)
log(event="done", best_val_micro=best, reference_val_micro=r0["val_micro"])
