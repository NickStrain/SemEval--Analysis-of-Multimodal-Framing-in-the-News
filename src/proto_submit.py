"""No-training submission with the original HF mCPT encoder (A0_proto).

Score = cosine(paragraph embedding, frame prototype), prototypes = mean L2-normed mCPT embedding
of all CV paragraphs carrying each frame (no gradient updates). Thresholds = tuned on the A0_proto
5-fold OOF scores (implicit None, global). Never touches the locked final split.

Usage: python -m src.proto_submit --langs bg,en,pt
"""
from __future__ import annotations

import argparse

import numpy as np
from transformers import AutoTokenizer

from src.data import LABELS, ROOT, load_cv
from src.decode import decode, tune
from src.final_model import load_dev
from src.train_cv import DEV, MODELS, Clf, embed, encode_inputs, targets

MODE = "implicit"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="bg,en,pt")
    a = ap.parse_args()
    cv = load_cv()
    Y = np.stack(cv.labels).astype(bool)
    T = tune(np.load(ROOT / "runs/phase1/A0_proto/oof.npz")["scores"][0], Y, MODE)

    enc, tok_name, pooling = MODELS["mcpt"]
    tok = AutoTokenizer.from_pretrained(tok_name)
    m = Clf(enc, pooling, 1, False).to(DEV)
    norm = lambda z: z / np.linalg.norm(z, axis=1, keepdims=True)
    Zn = norm(embed(m, encode_inputs(tok, cv)[0], tok.pad_token_id))
    Y15 = targets(cv, 15)
    P = norm(np.stack([Zn[Y15[:, j] == 1].mean(0) for j in range(15)]))

    sub = ROOT / "runs" / "submissions"; sub.mkdir(parents=True, exist_ok=True)
    for lang in a.langs.split(","):
        dev = load_dev(lang)
        S = norm(embed(m, encode_inputs(tok, dev)[0], tok.pad_token_id)) @ P.T
        pred = decode(S.astype(np.float32), T, MODE)
        path = sub / f"mcpt_proto_{lang.upper()}_dev.txt"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r, p in zip(dev.itertuples(), pred):
                f.write("\t".join([r.doc, r.start, r.end] + [LABELS[i] for i in np.flatnonzero(p)]) + "\n")
        print(f"wrote {path}: {len(dev)} paragraphs, {pred.sum(1).mean():.2f} labels/para, "
              f"{100 * (pred.sum(1) == 0).mean():.1f}% None")
