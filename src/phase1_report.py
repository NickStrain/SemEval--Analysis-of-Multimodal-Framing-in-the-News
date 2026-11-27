"""Phase 1 table: every run x None-mode x threshold scope, official micro/macro per fold, mean ± std.

Usage: python -m src.phase1_report [run_name ...]   (default: all runs/phase1/*/oof.npz)
Writes runs/phase1/results.json and runs/phase1/report.md
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.data import LANGS, ROOT, load_cv
from src.decode import crossfit
from src.eval import official_scores

P1 = ROOT / "runs" / "phase1"
df = load_cv()
Y = np.stack(df.labels).astype(bool)
FOLDS, LANG = df.fold.values, df.lang.values


def score_variant(job):
    name, mode, per_lang = job
    z = np.load(P1 / name / "oof.npz")
    scores, ckpts = z["scores"], list(z["ckpts"])
    if mode == "explicit" and np.isnan(scores[:, :, 14]).all():
        return None
    pred, chosen = crossfit(scores, Y, FOLDS, LANG, mode, per_lang)
    per_fold = []
    for k in range(5):
        m = FOLDS == k
        mi, ma = official_scores(Y[m], pred[m])
        rec = {"micro": mi, "macro": ma}
        for lg in LANGS:
            ml = m & (LANG == lg)
            rec[f"{lg}_micro"], rec[f"{lg}_macro"] = official_scores(Y[ml], pred[ml])
        per_fold.append(rec)
    return dict(run=name, mode=mode, thr="per-lang" if per_lang else "global",
                chosen=[ckpts[c] for c in chosen], per_fold=per_fold,
                mean={k: float(np.mean([r[k] for r in per_fold])) for k in per_fold[0]},
                std={k: float(np.std([r[k] for r in per_fold])) for k in per_fold[0]})


def fmt(r, key):
    return f"{100 * r['mean'][key]:.1f} ± {100 * r['std'][key]:.1f}"


if __name__ == "__main__":
    runs = sys.argv[1:] or sorted(p.parent.name for p in P1.glob("*/oof.npz"))
    jobs = [(n, m, pl) for n in runs for m in ("implicit", "explicit") for pl in (False, True)]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = [r for r in ex.map(score_variant, jobs) if r]
    out = P1 / "results.json"
    old = {(r["run"], r["mode"], r["thr"]): r for r in (json.loads(out.read_text()) if out.exists() else [])}
    old.update({(r["run"], r["mode"], r["thr"]): r for r in res})
    allres = sorted(old.values(), key=lambda r: (r["run"], r["mode"], r["thr"]))
    out.write_text(json.dumps(allres, indent=1))

    cols = ["micro", "macro"] + [f"{lg}_micro" for lg in LANGS] + [f"{lg}_macro" for lg in LANGS]
    lines = ["| run | None | thr | " + " | ".join(cols) + " | ckpt/fold |",
             "|" + "---|" * (len(cols) + 4)]
    for r in allres:
        lines.append(f"| {r['run']} | {r['mode']} | {r['thr']} | " + " | ".join(fmt(r, c) for c in cols)
                     + f" | {','.join(r['chosen'])} |")
    (P1 / "report.md").write_text("Official 15-class F1 (x100), mean ± std over 5 folds; "
                                  "thresholds & checkpoint cross-fitted on the other 4 folds.\n\n"
                                  + "\n".join(lines) + "\n")
    print("\n".join(lines))
