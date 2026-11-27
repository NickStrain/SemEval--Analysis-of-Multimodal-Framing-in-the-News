"""Evaluation for Subtask 1, matching starter-kit/.../scorer-subtask-1.py exactly:

  * an empty label set is scored as an explicit "None" class -> 15 classes;
  * sklearn f1_score(average="micro"/"macro", zero_division=1).

Note zero_division=1: a class with no gold and no predicted instances scores F1=1 in macro.
This matters for per-language subsets (rare labels can be absent).

Library:
    res = evaluate(y_true, y_pred, langs)   # (N,14) multi-hot arrays (no None column)
    micro, macro = official_scores(y_true, y_pred)

CLI (official file format: doc_id<TAB>start<TAB>end<TAB>labels...):
    python -m src.eval --gold gold.tsv --pred pred.tsv
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.data import LABEL2ID, LABELS

NONE_LABEL = "None"
LABELS15 = LABELS + [NONE_LABEL]


def add_none(y: np.ndarray) -> np.ndarray:
    """(N,14) -> (N,15) with None = 1 iff the row is empty."""
    y = np.asarray(y)[:, : len(LABELS)].astype(np.int8)
    return np.concatenate([y, (y.sum(1) == 0)[:, None].astype(np.int8)], axis=1)


def official_scores(y_true, y_pred) -> tuple[float, float]:
    """(micro_f1, macro_f1) exactly as the official scorer computes them."""
    t, p = add_none(y_true), add_none(y_pred)
    return (float(f1_score(t, p, average="micro", zero_division=1)),
            float(f1_score(t, p, average="macro", zero_division=1)))


def evaluate(y_true, y_pred, langs) -> dict:
    y_true, y_pred, langs = np.asarray(y_true), np.asarray(y_pred), np.asarray(langs)
    mi, ma = official_scores(y_true, y_pred)
    res = {"overall": dict(micro_f1=mi, macro_f1=ma, n=len(y_true)), "per_lang": {}, "per_label": {}}
    for lg in sorted(set(langs)):
        m = langs == lg
        mi, ma = official_scores(y_true[m], y_pred[m])
        res["per_lang"][lg] = dict(micro_f1=mi, macro_f1=ma, n=int(m.sum()))
    t, p = add_none(y_true), add_none(y_pred)
    P, R, F, S = precision_recall_fscore_support(t, p, average=None, zero_division=1)
    for i, lab in enumerate(LABELS15):
        res["per_label"][lab] = dict(precision=float(P[i]), recall=float(R[i]), f1=float(F[i]),
                                     support=int(S[i]), predicted=int(p[:, i].sum()))
    return res


def format_report(res: dict) -> str:
    o = res["overall"]
    out = [f"OVERALL  micro-F1 {o['micro_f1']:.4f}  macro-F1 {o['macro_f1']:.4f}  (n={o['n']}, 15 classes incl. None)",
           "", f"{'lang':<6}{'micro':>8}{'macro':>8}{'n':>7}"]
    for lg, v in res["per_lang"].items():
        out.append(f"{lg:<6}{v['micro_f1']:>8.4f}{v['macro_f1']:>8.4f}{v['n']:>7}")
    out += ["", f"{'label':<46}{'P':>7}{'R':>7}{'F1':>7}{'support':>9}{'pred':>7}"]
    for lab, v in res["per_label"].items():
        out.append(f"{lab:<46}{v['precision']:>7.3f}{v['recall']:>7.3f}{v['f1']:>7.3f}"
                   f"{v['support']:>9}{v['predicted']:>7}")
    return "\n".join(out)


# ---------- official file format I/O ----------

def read_official(path: str) -> dict[tuple[str, str, str], list[str]]:
    """Same tolerance as the official reader; later duplicates win."""
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split("\t")]
            out[tuple(parts[:3])] = [x for x in parts[3:] if x]
    return out


def write_official(path: str, keys, y: np.ndarray) -> None:
    """keys: iterable of (doc_id, start, end); y: (N,14) multi-hot."""
    with open(path, "w", encoding="utf-8") as f:
        for (d, s, e), row in zip(keys, y):
            f.write("\t".join([d, str(s), str(e)] + [LABELS[i] for i in np.flatnonzero(row[:14])]) + "\n")


def to_multihot(label_lists) -> np.ndarray:
    y = np.zeros((len(label_lists), len(LABELS)), dtype=np.int8)
    for i, labs in enumerate(label_lists):
        for l in labs:
            y[i, LABEL2ID[l]] = 1
    return y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    a = ap.parse_args()
    gold, pred = read_official(a.gold), read_official(a.pred)
    if set(gold) != set(pred):
        raise SystemExit(f"key mismatch: {len(set(gold) ^ set(pred))} rows differ")
    keys = list(gold)
    langs = [k[0][:2].lower() for k in keys]
    res = evaluate(to_multihot([gold[k] for k in keys]), to_multihot([pred[k] for k in keys]), langs)
    print(format_report(res))
