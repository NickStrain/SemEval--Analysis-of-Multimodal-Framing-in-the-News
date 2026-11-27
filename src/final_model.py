"""Train one model on ALL CV data (never the locked final split), predict a dev set, write a submission.

  * config      = a CV run (runs/<phase>/<name>/config.json) -> same model/inputs/extra data/lr
  * epoch       = median of the epochs chosen per fold in CV (cross-fitted, None=implicit, global thr)
  * thresholds  = tuned on all 5 folds' OOF scores at that epoch (global, implicit None)
  * dev paragraphs are aligned to raw-document blocks by order (verified on all 348 training docs;
    the official character offsets are inconsistent), and start/end are copied verbatim to the output.

Usage: python -m src.final_model --run phase1b/M_E1_extraEN --lang en
Outputs: runs/submissions/<run>_<LANG>_dev.txt, ckpt/<run>.pt
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data import DATA, LABELS, ROOT, load_cv, load_extra
from src.decode import crossfit, decode, tune
from src.train_cv import DEV, MODELS, Clf, encode_inputs, pad, seed_all, targets

MODE = "implicit"


def load_dev(lang: str) -> pd.DataFrame:
    d = DATA / "3_dev-set-nov-release" / lang.upper()
    rows = []
    with open(d / "subtask-1-participant.txt", encoding="utf-8-sig") as f:
        for line in f:
            p = [x.strip() for x in line.rstrip("\r\n").split("\t")]
            if len(p) >= 3 and p[0]:
                rows.append(dict(doc=p[0], start=p[1], end=p[2]))
    df = pd.DataFrame(rows)
    out = []
    for doc, g in df.groupby("doc", sort=False):
        raw = (d / "raw-documents" / doc).read_text(encoding="utf-8").replace("\r\n", "\n")
        blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
        g = g.sort_values("start", key=lambda s: s.astype(int))
        if len(blocks) != len(g):
            raise ValueError(f"{doc}: {len(blocks)} raw blocks vs {len(g)} paragraphs")
        for i, r in enumerate(g.itertuples()):
            out.append(dict(doc=doc, start=r.start, end=r.end, lang=lang, title=blocks[0], text=blocks[i],
                            prev_text=blocks[i - 1] if i else "", next_text=blocks[i + 1] if i + 1 < len(blocks) else ""))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="e.g. phase1b/M_E1_extraEN")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--exclude-pools", action="store_true",
                    help="drop sr/pools.json PRACTICE+RESERVED articles from training and threshold tuning")
    ap.add_argument("--no-dev", action="store_true", help="only train + save the checkpoint")
    ap.add_argument("--tag", default=None, help="checkpoint name (default: run name)")
    a = ap.parse_args()
    run_dir = ROOT / "runs" / a.run
    cfg = SimpleNamespace(**json.loads((run_dir / "config.json").read_text()))
    cv = load_cv()
    excl = set()
    if a.exclude_pools:
        from sr.pools import load_pools
        pools = load_pools(); excl = set(pools["practice"]) | set(pools["reserved"])
    keep = ~cv.article_id.isin(excl).values
    Y = np.stack(cv.labels).astype(bool)

    # 1) epoch + thresholds from the CV out-of-fold scores
    oof = np.load(run_dir / "oof.npz")["scores"]
    _, chosen = crossfit(oof[:, keep], Y[keep], cv.fold.values[keep], cv.lang.values[keep], MODE, False)
    ep = int(statistics.median_low([c + 1 for c in chosen]))
    T = tune(oof[ep - 1][keep], Y[keep], MODE)  # pool rows never touch the thresholds
    cv = cv[keep].reset_index(drop=True)
    print(f"training rows (Task 9): {len(cv)}  excluded articles: {len(excl)}")
    print(f"CV-chosen epochs per fold {[c + 1 for c in chosen]} -> train {ep} epochs (schedule {cfg.epochs})")

    # 2) train on all CV paragraphs (+ the run's extra data)
    enc, tok_name, pooling = MODELS[cfg.model]
    tok = AutoTokenizer.from_pretrained(tok_name)
    extra = load_extra(tuple(cfg.extra.split(","))) if getattr(cfg, "extra", "") else None
    full = pd.concat([cv, extra], ignore_index=True) if extra is not None else cv
    tgt, ctx = encode_inputs(tok, full)
    Yt = targets(full, cfg.n_out)
    W = np.ones_like(Yt)
    if extra is not None and cfg.n_out == 15:
        W[len(cv) + np.flatnonzero(extra.mask_none.values.astype(bool)), 14] = 0.0
    langs = full.lang.values
    en_rows = np.flatnonzero((langs == "en") & (np.arange(len(full)) < len(cv)))
    rows = np.concatenate([np.arange(len(full))] + [en_rows] * (getattr(cfg, "en_over", 1) - 1))
    seed_all(getattr(cfg, "seed", 13))
    model = Clf(enc, pooling, cfg.n_out, cfg.ctx).to(DEV)
    if cfg.freeze_emb:
        model.enc.embeddings.word_embeddings.weight.requires_grad_(False)
    if getattr(cfg, "grad_ckpt", False):  # same memory setting as the CV run (needed for XLM-R large on 8 GB)
        model.enc.gradient_checkpointing_enable()
    opt = torch.optim.AdamW([{"params": [p for p in model.enc.parameters() if p.requires_grad], "lr": cfg.lr},
                             {"params": model.head.parameters(), "lr": cfg.head_lr}], weight_decay=0.01)
    spe = math.ceil(len(rows) / cfg.bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * spe * cfg.epochs), spe * cfg.epochs)

    def inputs(idx, T_, C_):
        ids, mask = pad([T_[i] for i in idx], tok.pad_token_id)
        kw = dict(zip(("cids", "cmask"), pad([C_[i] for i in idx], tok.pad_token_id))) if cfg.ctx else {}
        return ids, mask, kw

    for e in range(ep):
        model.train(); t0 = time.time()
        perm = np.random.permutation(rows)
        for b in range(0, len(perm), cfg.bs):
            idx = perm[b: b + cfg.bs]
            ids, mask, kw = inputs(idx, tgt, ctx)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, mask, **kw)
            w = torch.from_numpy(W[idx]).to(DEV)
            l = nn.functional.binary_cross_entropy_with_logits(logits.float(), torch.from_numpy(Yt[idx]).to(DEV), reduction="none")
            loss = (l * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        print(json.dumps(dict(epoch=e + 1, sec=round(time.time() - t0, 1))), flush=True)
    (ROOT / "ckpt").mkdir(exist_ok=True)
    tag = a.tag or a.run.replace("/", "_")
    torch.save({"state_dict": model.state_dict(), "config": vars(cfg), "epochs": ep, "thresholds": T.tolist(),
                "mode": MODE}, ROOT / "ckpt" / f"{tag}.pt")

    print(f"saved ckpt/{tag}.pt")
    if a.no_dev:
        raise SystemExit
    # 3) predict dev
    dev = load_dev(a.lang)
    dt, dc = encode_inputs(tok, dev)
    model.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(dev), 64):
            ids, mask, kw = inputs(range(b, min(b + 64, len(dev))), dt, dc)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(torch.sigmoid(model(ids, mask, **kw).float()).cpu().numpy())
    S = np.full((len(dev), 15), np.nan, np.float32)
    S[:, : cfg.n_out] = np.concatenate(out)
    P = decode(S, T, MODE)
    sub = ROOT / "runs" / "submissions"; sub.mkdir(parents=True, exist_ok=True)
    path = sub / f"{tag}_{a.lang.upper()}_dev.txt"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r, p in zip(dev.itertuples(), P):
            f.write("\t".join([r.doc, r.start, r.end] + [LABELS[i] for i in np.flatnonzero(p)]) + "\n")
    np.save(sub / f"{tag}_{a.lang.upper()}_dev_scores.npy", S)
    print(f"wrote {path}: {len(dev)} paragraphs, {P.sum(1).mean():.2f} labels/para, "
          f"{100 * (P.sum(1) == 0).mean():.1f}% None")
