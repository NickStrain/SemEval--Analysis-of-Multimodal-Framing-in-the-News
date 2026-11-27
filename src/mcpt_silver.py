"""Silver adaptation of the ORIGINAL mCPT encoder (continued training, not from scratch).

Data  : data/silver/silver_en.parquet (150K mm-framing article summaries + article-level frames, see src/silver.py).
Input : "title [SEP] summary", max 128 tokens; mean pooling; 14-way BCE head (discarded afterwards).
Train : 1 epoch (or --epochs), batch 64, lr 5e-5 (head 1e-3), warmup 5%, bf16 autocast.
Saves : ckpt/mcpt_silver.pt  = encoder state_dict only (gold stage re-initialises its own head).
Usage : python -m src.mcpt_silver [--epochs 1]
"""
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data import ROOT
from src.train_cv import DEV, MODELS, Clf, pad, seed_all

ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--out", default="ckpt/mcpt_silver.pt"); a = ap.parse_args()
MAXLEN, BS = 128, 64
seed_all(0)
enc, tok_name, pooling = MODELS["mcpt"]
tok = AutoTokenizer.from_pretrained(tok_name)
df = pd.read_parquet(ROOT / "data/silver/silver_en.parquet")
seqs = tok(df.title.tolist(), df.text.tolist(), truncation="only_second", max_length=MAXLEN)["input_ids"]
Y = np.stack(df.y).astype(np.float32)
model = Clf(enc, pooling, 14, False).to(DEV)
opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": 5e-5},
                         {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
spe = math.ceil(len(seqs) / BS); steps = spe * a.epochs
sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
t0 = time.time(); run = 0.0; s = 0
for ep in range(a.epochs):
    model.train(); perm = np.random.permutation(len(seqs))
    for b in range(spe):
        idx = perm[b * BS:(b + 1) * BS]
        ids, m = pad([seqs[i] for i in idx], tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, m).float()
        loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(Y[idx]).to(DEV))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        run = 0.98 * run + 0.02 * loss.item() if s else loss.item(); s += 1
        if s % 500 == 0 or s == steps:
            print(json.dumps(dict(step=s, of=steps, loss=round(run, 4), min=round((time.time() - t0) / 60, 1))), flush=True)
torch.save(model.enc.state_dict(), ROOT / a.out); print(f"saved {a.out}")
