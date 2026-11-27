"""5-fold CV training; saves out-of-fold scores for every epoch / checkpoint.

Output: runs/phase1/<name>/oof.npz with
  scores (C, N, 15) float32  OOF scores per checkpoint (NaN in col 14 if --n-out 14)
  ckpts  (C,)                checkpoint names
rows in load_cv() order.

Configs:
  --kind proto    A0: cosine to label prototypes (mean of L2-normed mCPT embeddings), no training
  --kind linear   A : frozen mCPT embeddings + one-vs-rest logistic regression (checkpoints = C grid)
  --kind finetune B/C: full fine-tuning, encoder + linear head, BCE (checkpoints = epochs)
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

import pandas as pd

from src.data import ROOT, load_cv, load_extra

MODELS = {
    "mpnet": ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
              "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "mean"),
    "mcpt": ("socialcomplab/mcpt-body-semeval2023task3",
             "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "mean"),
    "xlmr-base": ("FacebookAI/xlm-roberta-base", "FacebookAI/xlm-roberta-base", "cls"),
    "xlmr-large": ("FacebookAI/xlm-roberta-large", "FacebookAI/xlm-roberta-large", "cls"),
}
TGT_LEN = 256
CTX_LENS = (48, 100, 100)  # title, prev, next (tokens, before specials) -> <= 256 total
DEV = "cuda"


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def encode_inputs(tok, df):
    tgt = tok(df.text.tolist(), truncation=True, max_length=TGT_LEN)["input_ids"]
    pieces = [tok(df[c].tolist(), add_special_tokens=False)["input_ids"] for c in ("title", "prev_text", "next_text")]
    ctx = []
    for t, p, n in zip(*pieces):
        ids = [tok.cls_token_id]
        for seq, L in zip((t, p, n), CTX_LENS):
            ids += seq[:L] + [tok.sep_token_id]
        ctx.append(ids)
    return tgt, ctx


def pad(seqs, pad_id):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s); mask[i, : len(s)] = 1
    return ids.to(DEV), mask.to(DEV)


class Clf(nn.Module):
    def __init__(self, enc_name, pooling, n_out, ctx):
        super().__init__()
        self.enc = AutoModel.from_pretrained(enc_name)
        self.pooling, self.ctx = pooling, ctx
        h = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h * (2 if ctx else 1), n_out))

    def pool(self, ids, mask):
        out = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        if self.pooling == "cls":
            return out[:, 0]
        m = mask.unsqueeze(-1).to(out.dtype)
        return (out * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, ids, mask, cids=None, cmask=None, return_z=False):
        z = self.pool(ids, mask)
        if return_z:
            zt = z
            if self.ctx:
                z = torch.cat([z, self.pool(cids, cmask)], -1)
            return self.head(z), zt
        if self.ctx:
            z = torch.cat([z, self.pool(cids, cmask)], -1)
        return self.head(z)


def label_contrastive(z, y, tau=0.1):
    """Multi-label label-aware contrastive loss (mCPT-style): positives weighted by label-set Jaccard similarity
    (y includes the None column, so None paragraphs attract each other). Anchors without any positive are skipped."""
    z = nn.functional.normalize(z.float(), dim=-1)
    y = y.float(); inter = y @ y.T; union = y.sum(1, keepdim=True) + y.sum(1) - inter
    w = torch.where(union > 0, inter / union.clamp(min=1), torch.zeros_like(inter))
    eye = torch.eye(len(z), device=z.device, dtype=torch.bool)
    w = w.masked_fill(eye, 0)
    logits = (z @ z.T / tau).masked_fill(eye, -1e9)
    logp = logits - torch.logsumexp(logits, 1, keepdim=True)
    ok = w.sum(1) > 0
    if not ok.any():
        return z.sum() * 0
    return (-(w * logp).sum(1)[ok] / w.sum(1)[ok]).mean()


@torch.no_grad()
def embed(model: Clf, seqs, pad_id, bs=128):
    model.eval(); out = []
    order = np.argsort([len(s) for s in seqs])
    for i in range(0, len(seqs), bs):
        idx = order[i: i + bs]
        ids, mask = pad([seqs[j] for j in idx], pad_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model.pool(ids, mask).float().cpu())
    z = torch.cat(out).numpy()
    res = np.empty_like(z); res[order] = z
    return res


def targets(df, n_out):
    Y = np.stack(df.labels).astype(np.float32)
    if n_out == 15:
        Y = np.concatenate([Y, (Y.sum(1) == 0)[:, None].astype(np.float32)], 1)
    return Y


# ---------------- A0 / A ----------------

def run_frozen(args, df, out_dir):
    enc, tok_name, pooling = MODELS["mcpt"]
    tok = AutoTokenizer.from_pretrained(tok_name)
    m = Clf(enc, pooling, 1, False).to(DEV)
    tgt, ctx = encode_inputs(tok, df)
    Z = embed(m, tgt, tok.pad_token_id)
    if args.ctx:
        Z = np.concatenate([Z, embed(m, ctx, tok.pad_token_id)], 1)
    Y15 = targets(df, 15)
    folds = df.fold.values
    if args.kind == "proto":
        Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        scores = np.zeros((1, len(df), 15), np.float32)
        for k in range(5):
            tr, te = folds != k, folds == k
            P = np.stack([Zn[tr][Y15[tr, j] == 1].mean(0) for j in range(15)])
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            scores[0, te] = Zn[te] @ P.T
        return scores, ["proto"]
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Cs = [0.01, 0.1, 1.0]
    scores = np.zeros((len(Cs), len(df), 15), np.float32)
    for k in range(5):
        tr, te = folds != k, folds == k
        sc = StandardScaler().fit(Z[tr])
        Xtr, Xte = sc.transform(Z[tr]), sc.transform(Z[te])
        for ci, C in enumerate(Cs):
            for j in range(15):
                lr = LogisticRegression(C=C, max_iter=3000).fit(Xtr, Y15[tr, j])
                scores[ci, te, j] = lr.predict_proba(Xte)[:, 1]
        print(f"fold {k} done", flush=True)
    return scores, [f"C={c}" for c in Cs]


# ---------------- B / C ----------------

def run_finetune(args, df, out_dir):
    """Rows 0..N-1 = Task-9 CV paragraphs (scored); rows N.. = extra training-only paragraphs.

    --extra en[,ru]   add batch-1 paragraphs to every fold's training set; their None output is masked
    --en-over K       Task-9 EN training paragraphs appear K times per epoch
    --stage2-en E     after the main epochs, continue E epochs on English only (Task-9 EN train + extra EN),
                      lr x0.5; EN test rows are scored after each such epoch -> oof_stage2.npz
    """
    enc, tok_name, pooling = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(tok_name)
    N = len(df)
    extra = load_extra(tuple(args.extra.split(","))) if args.extra else None
    if getattr(args, "trans", ""):  # machine-translated gold (src_fold = CV fold of the source article, -1 otherwise)
        tr_df = pd.read_parquet(ROOT / args.trans)
        tr_df["labels"] = [np.asarray(y, np.int8) for y in tr_df.labels]
        extra = pd.concat([extra, tr_df], ignore_index=True) if extra is not None else tr_df
    if extra is not None:
        extra["src_fold"] = extra["src_fold"].fillna(-1).astype(int) if "src_fold" in extra else -1
    full = pd.concat([df, extra], ignore_index=True) if extra is not None else df
    if getattr(args, "artframes", ""):  # Step 8: prepend OOF-predicted article frames to the target paragraph
        af = json.load(open(ROOT / args.artframes))
        pid = (full.article_id + "#" + full.para_idx.astype("Int64").astype(str)).tolist()
        full = full.copy(); full["text"] = [af.get(p, "") + t for p, t in zip(pid, full.text)]
    tgt, ctx = encode_inputs(tok, full)
    Y = targets(full, args.n_out)
    W = np.ones_like(Y)
    if extra is not None and args.n_out == 15:
        W[N + np.flatnonzero(extra.mask_none.values.astype(bool)), 14] = 0.0  # batch-1 lists only framed paragraphs: never teach the None output from them
    langs = full.lang.values
    folds = df.fold.values
    x_idx = np.arange(N, len(full))
    scores = np.full((args.epochs, N, 15), np.nan, np.float32)
    scores2 = np.full((max(args.stage2_en, 1), N, 15), np.nan, np.float32)
    log = []

    def batch_inputs(idx):
        ids, mask = pad([tgt[i] for i in idx], tok.pad_token_id)
        kw = dict(zip(("cids", "cmask"), pad([ctx[i] for i in idx], tok.pad_token_id))) if args.ctx else {}
        return ids, mask, kw

    Ycl = targets(full, 15)  # label sets incl. None for the contrastive term

    def train_epoch(model, opt, sched, rows, lam=0.0, cl_only=False):
        model.train(); tot = 0.0
        perm = np.random.permutation(rows)
        for b in range(0, 20 * args.bs if args.smoke else len(perm), args.bs):
            idx = perm[b: b + args.bs]
            ids, mask, kw = batch_inputs(idx)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, z = model(ids, mask, **kw, return_z=True)
            cl = label_contrastive(z, torch.from_numpy(Ycl[idx]).to(DEV), args.tau) if (lam or cl_only) else 0.0
            if cl_only:
                loss = cl
            else:
                w = torch.from_numpy(W[idx]).to(DEV)
                l = nn.functional.binary_cross_entropy_with_logits(logits.float(), torch.from_numpy(Y[idx]).to(DEV),
                                                                   reduction="none")
                loss = (l * w).sum() / w.sum() + lam * cl
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item() * len(idx)
        return tot / len(rows)

    @torch.no_grad()
    def predict(model, rows):
        model.eval(); out = []
        order = rows[np.argsort([len(tgt[i]) for i in rows])]
        for b in range(0, len(order), 64):
            ids, mask, kw = batch_inputs(order[b: b + 64])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(torch.sigmoid(model(ids, mask, **kw).float()).cpu().numpy())
        return order, np.concatenate(out)

    def make_opt(model, lr_scale, n_steps, warm):
        enc_params = [p for p in model.enc.parameters() if p.requires_grad]
        opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr * lr_scale},
                                 {"params": model.head.parameters(), "lr": args.head_lr * lr_scale}],
                                weight_decay=0.01)
        return opt, get_linear_schedule_with_warmup(opt, int(warm * n_steps), n_steps)

    for k in range(1 if args.smoke else 5):
        seed_all(args.seed + k)
        tr_idx, te_idx = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        xk = x_idx[extra.src_fold.values[x_idx - N] != k] if extra is not None else x_idx  # no translation of a test article
        if getattr(args, "en_only", False):  # English text only: EN gold + English extra rows (incl. translations)
            tr_idx, xk = tr_idx[langs[tr_idx] == "en"], xk[langs[xk] == "en"]
        tr_en = tr_idx[langs[tr_idx] == "en"]
        rows = np.concatenate([tr_idx] + [tr_en] * (args.en_over - 1) + [xk])
        model = Clf(enc, pooling, args.n_out, args.ctx).to(DEV)
        if getattr(args, "init_enc", ""):
            model.enc.load_state_dict(torch.load(ROOT / args.init_enc, map_location=DEV))
        if args.freeze_emb:
            model.enc.embeddings.word_embeddings.weight.requires_grad_(False)
        if args.grad_ckpt:
            model.enc.gradient_checkpointing_enable()
        if args.supcon_pre:  # Phase A: label-aware contrastive continued pre-training of the encoder only
            opt, sched = make_opt(model, 1.0, math.ceil(len(rows) / args.bs) * args.supcon_pre, 0.1)
            for ep in range(args.supcon_pre):
                t0 = time.time(); tl = train_epoch(model, opt, sched, rows, cl_only=True)
                print(json.dumps(dict(fold=k, supcon_epoch=ep + 1, loss=tl, sec=time.time() - t0)), flush=True)
        opt, sched = make_opt(model, 1.0, math.ceil(len(rows) / args.bs) * args.epochs, 0.1)
        torch.cuda.reset_peak_memory_stats()
        for ep in range(args.epochs):
            t0 = time.time()
            tl = train_epoch(model, opt, sched, rows, lam=args.supcon)
            order, p = predict(model, te_idx)
            scores[ep, order, : args.n_out] = p
            rec = dict(fold=k, epoch=ep + 1, train_loss=tl, n_train=len(rows), sec=time.time() - t0,
                       peak_gb=torch.cuda.max_memory_allocated() / 2**30)
            log.append(rec); print(json.dumps(rec), flush=True)
        if args.stage2_en:
            rows2 = np.concatenate([tr_en, xk[langs[xk] == "en"]])
            te_en = te_idx[langs[te_idx] == "en"]
            opt, sched = make_opt(model, 0.5, math.ceil(len(rows2) / args.bs) * args.stage2_en, 0.0)
            for ep in range(args.stage2_en):
                t0 = time.time()
                tl = train_epoch(model, opt, sched, rows2)
                order, p = predict(model, te_en)
                scores2[ep, order, : args.n_out] = p
                rec = dict(fold=k, stage2_epoch=ep + 1, train_loss=tl, n_train=len(rows2), sec=time.time() - t0)
                log.append(rec); print(json.dumps(rec), flush=True)
        del model, opt; torch.cuda.empty_cache()
    (out_dir / "train_log.jsonl").write_text("\n".join(json.dumps(r) for r in log))
    if args.stage2_en:
        np.savez_compressed(out_dir / "oof_stage2.npz", scores=scores2,
                            ckpts=np.array([f"s2ep{e + 1}" for e in range(args.stage2_en)]))
    return scores, [f"ep{e + 1}" for e in range(args.epochs)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--phase", default="phase1", help="output folder under runs/")
    ap.add_argument("--kind", choices=["proto", "linear", "finetune"], required=True)
    ap.add_argument("--model", default="mcpt", choices=list(MODELS))
    ap.add_argument("--ctx", action="store_true")
    ap.add_argument("--n-out", type=int, default=15, choices=[14, 15])
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--freeze-emb", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--extra", default="", help="batch-1 extra training data, e.g. en or en,ru")
    ap.add_argument("--en-over", type=int, default=1, help="repeat Task-9 EN training paragraphs K times/epoch")
    ap.add_argument("--stage2-en", type=int, default=0, help="extra English-only epochs after main training")
    ap.add_argument("--artframes", default="", help="json pid -> '[frames: ...] ' prefix (Step 8)")
    ap.add_argument("--smoke", action="store_true", help="fold 0, 20 steps/epoch: speed+memory check")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--init-enc", default="", help="encoder state_dict to start from (e.g. ckpt/mcpt_silver.pt)")
    ap.add_argument("--supcon", type=float, default=0.0, help="weight of the label-aware contrastive term during BCE training")
    ap.add_argument("--supcon-pre", type=int, default=0, help="contrastive-only epochs before BCE training (Phase A)")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--en-only", action="store_true", help="train on English text only (EN gold + EN extra/translations)")
    ap.add_argument("--trans", default="", help="parquet of machine-translated gold rows (extra training only)")
    args = ap.parse_args()
    out_dir = ROOT / "runs" / args.phase / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_cv()
    t0 = time.time()
    scores, ckpts = (run_frozen if args.kind in ("proto", "linear") else run_finetune)(args, df, out_dir)
    np.savez_compressed(out_dir / "oof.npz", scores=scores, ckpts=np.array(ckpts))
    (out_dir / "config.json").write_text(json.dumps(vars(args) | {"minutes": (time.time() - t0) / 60}, indent=1))
    print(f"saved {out_dir} in {(time.time() - t0) / 60:.1f} min")
