"""Step 6.3a: DeBERTa-v3-large, 1 epoch on silver (title [SEP] [SEP] summary), BCE. Saves ckpt/deb_silver.pt.
Batch 32 (no accumulation), max 128 tokens, lr 2e-5, warmup 5%, bf16 autocast, frozen word embeddings."""
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data import ROOT
from src.deberta_cv import DEV, MODEL, Net, pad, seed_all

MAXLEN, BS, LR = 128, 32, 2e-5
seed_all(0)
tok = AutoTokenizer.from_pretrained(MODEL)
df = pd.read_parquet(ROOT / "data/silver/silver_en.parquet")
T = tok(df.title.tolist(), add_special_tokens=False)["input_ids"]; X = tok(df.text.tolist(), add_special_tokens=False)["input_ids"]
seqs = []
for t, x in zip(T, X):
    t = t[:32]; seqs.append([tok.cls_token_id] + t + [tok.sep_token_id, tok.sep_token_id] + x[:MAXLEN - 4 - len(t)] + [tok.sep_token_id])
Y = np.stack(df.y).astype(np.float32)
net = Net().to(DEV); net.enc.embeddings.word_embeddings.weight.requires_grad_(False); net.enc.gradient_checkpointing_enable()
opt = torch.optim.AdamW([{"params": [p for p in net.enc.parameters() if p.requires_grad], "lr": LR},
                         {"params": net.head.parameters(), "lr": LR * 10}], weight_decay=0.01)
steps = math.ceil(len(seqs) / BS); sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
perm = np.random.permutation(len(seqs)); t0 = time.time(); run = 0.0
net.train()
for s in range(steps):
    idx = perm[s * BS:(s + 1) * BS]; ids, m = pad([seqs[i] for i in idx], tok.pad_token_id)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = net(ids, m).float()
    loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(Y[idx]).to(DEV))
    loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
    run = 0.98 * run + 0.02 * loss.item() if s else loss.item()
    if s % 500 == 0 or s == steps - 1:
        print(json.dumps(dict(step=s, of=steps, loss=round(run, 4), min=round((time.time() - t0) / 60, 1))), flush=True)
torch.save(net.state_dict(), ROOT / "ckpt" / "deb_silver.pt"); print("saved ckpt/deb_silver.pt")
