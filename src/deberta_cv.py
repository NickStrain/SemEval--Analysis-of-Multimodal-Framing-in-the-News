"""Step 4: DeBERTa-v3-large, English only, 5-fold CV on the EN folds (+ batch-1 EN as extra training rows).

Input  : "[CLS] title [SEP] prev_par (<=96 tok) [SEP] target_par [SEP]", max 384 tokens.
Train  : bf16 autocast, batch 4 x accumulation 8, gradient checkpointing, frozen word embeddings,
         lr 1e-5 (head 1e-4), warmup 10%, linear decay, 5 epochs. Loss: BCE or ASL (gamma_neg=4, clip 0.05).
Output : runs/step4/<name>/scores.npz
           oof (E, N_en, 14)  test-fold probabilities after each epoch (NaN where not run)
           dev (5, E, N_dev, 14) dev probabilities of each fold model after each epoch
Usage  : python -m src.deberta_cv --name deb_bce --loss bce [--folds 0] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.data import ROOT, load_extra
from src.en import load_en
from src.final_model import load_dev

MODEL = "microsoft/deberta-v3-large"
DEV = "cuda"
MAXLEN, PREV, TITLE = 384, 96, 48


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build(tok, titles, prevs, texts):
    enc = lambda xs: tok(list(xs), add_special_tokens=False)["input_ids"]
    T, P, X = enc(titles), enc(prevs), enc(texts)
    out = []
    for t, p, x in zip(T, P, X):
        t, p = t[:TITLE], p[:PREV]
        budget = MAXLEN - 4 - len(t) - len(p)
        out.append([tok.cls_token_id] + t + [tok.sep_token_id] + p + [tok.sep_token_id] + x[:budget] + [tok.sep_token_id])
    return out


def pad(seqs, pad_id):
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long); m = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s); m[i, :len(s)] = 1
    return ids.to(DEV), m.to(DEV)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL, dtype=torch.float32)  # config says fp16 -> NaN with AdamW
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.enc.config.hidden_size, 14))

    def forward(self, ids, mask):
        return self.head(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])


def asl(logits, y, gn=4.0, clip=0.05):
    p = torch.sigmoid(logits); pn = (p - clip).clamp(min=0)
    lp = y * torch.log(p.clamp(min=1e-8))
    ln = (1 - y) * torch.log((1 - pn).clamp(min=1e-8)) * pn.pow(gn)
    return -(lp + ln).mean()


@torch.no_grad()
def predict(model, seqs, pad_id, bs=16):
    model.eval(); order = np.argsort([len(s) for s in seqs]); out = np.zeros((len(seqs), 14), np.float32)
    for b in range(0, len(seqs), bs):
        idx = order[b:b + bs]; ids, m = pad([seqs[i] for i in idx], pad_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[idx] = torch.sigmoid(model(ids, m).float()).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--loss", choices=["bce", "asl"], default="bce")
    ap.add_argument("--folds", default="0,1,2,3,4"); ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--init", default="", help="state_dict to start from (e.g. ckpt/deb_silver.pt)")
    a = ap.parse_args()
    out_dir = ROOT / "runs" / "step4" / a.name; out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    en, Y, F = load_en(); ex = load_extra(("en",)); dev = load_dev("en")
    X = build(tok, en.title, en.prev_text, en.text)
    Xx = build(tok, ex.title, ex.prev_text, ex.text); Yx = np.stack(ex.labels).astype(np.float32)
    Xd = build(tok, dev.title, dev.prev_text, dev.text)
    allX = X + Xx; allY = np.concatenate([Y.astype(np.float32), Yx])
    f_ = out_dir / "scores.npz"
    if f_.exists():
        z = np.load(f_); oof, devp = z["oof"], z["dev"]
    else:
        oof = np.full((a.epochs, len(en), 14), np.nan, np.float32); devp = np.full((5, a.epochs, len(dev), 14), np.nan, np.float32)
    log = []
    for k in [int(x) for x in a.folds.split(",")]:
        seed_all(a.seed + k)
        tr = np.concatenate([np.flatnonzero(F != k), len(X) + np.arange(len(Xx))]); te = np.flatnonzero(F == k)
        model = Net().to(DEV)
        if a.init:
            model.load_state_dict(torch.load(ROOT / a.init, map_location=DEV))
        model.enc.embeddings.word_embeddings.weight.requires_grad_(False)
        model.enc.gradient_checkpointing_enable()
        opt = torch.optim.AdamW([{"params": [p for p in model.enc.parameters() if p.requires_grad], "lr": a.lr},
                                 {"params": model.head.parameters(), "lr": a.lr * 10}], weight_decay=0.01)
        bs, acc = 4, 8
        steps = math.ceil(len(tr) / (bs * acc)) * a.epochs
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        torch.cuda.reset_peak_memory_stats()
        for ep in range(a.epochs):
            model.train(); t0 = time.time(); perm = np.random.permutation(tr); tot = 0.0
            n_batches = 20 * acc if a.smoke else math.ceil(len(perm) / bs)
            for i in range(n_batches):
                idx = perm[i * bs:(i + 1) * bs]
                if len(idx) == 0:
                    break
                ids, m = pad([allX[j] for j in idx], tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(ids, m).float()
                y = torch.from_numpy(allY[idx]).to(DEV)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y) if a.loss == "bce" else asl(logits, y)
                (loss / acc).backward(); tot += loss.item()
                if (i + 1) % acc == 0 or i == n_batches - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            oof[ep, te] = predict(model, [X[j] for j in te], tok.pad_token_id)
            devp[k, ep] = predict(model, Xd, tok.pad_token_id)
            rec = dict(fold=k, epoch=ep + 1, loss=tot / n_batches, sec=round(time.time() - t0, 1),
                       peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
            log.append(rec); print(json.dumps(rec), flush=True)
            np.savez_compressed(f_, oof=oof, dev=devp)
            if a.smoke:
                break
        del model, opt; torch.cuda.empty_cache()
    with open(out_dir / "train_log.jsonl", "a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in log))


if __name__ == "__main__":
    main()
