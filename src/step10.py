"""Step 10: pick the best EN system by nested CV over every available model, then write the dev submission.

Models (EN OOF probabilities (907,14) + EN dev probabilities (558,14)):
  * train_cv runs (multilingual): OOF of fold k = epoch cross-fitted on the other folds (as oof/*_all.npy);
    dev = runs/submissions/<run>_EN_dev_scores.npy from src.final_model (trained on all CV data).
  * deberta_cv runs (EN only): OOF of fold k = epoch cross-fitted on the other folds; dev = mean over the
    5 fold models at their chosen epochs.
Candidates = every subset (size 1..3) of available models, weighted average (weights + per-label thresholds
nested: chosen on 4 folds, applied to the 5th), plus the Step-9 LightGBM stacker over the best subset.
Best = highest nested mean micro. Final weights/thresholds are then fit on all OOF and applied to dev.

Usage: python -m src.step10 [--write]
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics

import numpy as np

from src.data import LABELS, ROOT, load_cv
from src.decode import crossfit, decode, tune
from src.en import load_en, per_fold
from src.ens import fit_dec, weight_grid
from src.final_model import load_dev
from src.post import _micro

TRAIN_CV = {"mcpt": "phase1b/M_E1_extraEN", "xlmrb": "phase1/C_xlmr_base",
            "xlmrl": "phase1c/XL_E1", "xlmrb_x": "phase1c/XB_E1", "xlmrb_af": "phase1c/XB_E1_AF", "xlmrl_s2": "phase1c/XL_E1_s2"}
DEBERTA = ["deb_bce", "deb_silver_bce", "deb_silver_e8"]


def load_train_cv(run, en_mask, F_all, Y_all, langs):
    z = np.load(ROOT / "runs" / run / "oof.npz")["scores"]
    _, chosen = crossfit(z, Y_all, F_all, langs, "implicit", False)
    oof = np.zeros((len(F_all), 14), np.float32)
    for k, c in enumerate(chosen):
        oof[F_all == k] = z[c, F_all == k, :14]
    load_train_cv.full = oof
    tag = run.replace("/", "_")
    dev_f = ROOT / "runs" / "submissions" / f"{tag}_EN_dev_scores.npy"
    if run == "phase1b/M_E1_extraEN" and not dev_f.exists():
        dev_f = ROOT / "dev" / "mcpt_en.npy"
    dev = np.load(dev_f)[:, :14] if dev_f.exists() else None
    return oof[en_mask], dev, chosen


def load_deberta(name, Y, F):
    z = np.load(ROOT / "runs" / "step4" / name / "scores.npz")
    oof_e, dev_e = z["oof"], z["dev"]
    if np.isnan(oof_e[-1]).any():
        return None, None, None
    oof = np.zeros(Y.shape, np.float32); chosen = []
    for k in range(5):
        tr, te = F != k, F == k
        best = max(range(len(oof_e)), key=lambda e: _micro(Y[tr], oof_e[e][tr] >= fit_dec(oof_e[e][tr], Y[tr], "label")))
        chosen.append(best); oof[te] = oof_e[best][te]
    dev = np.mean([dev_e[k, chosen[k]] for k in range(5)], 0)
    return oof, dev, chosen


FINE = np.round(np.arange(0.01, 0.951, 0.01), 2)


def fit(S, Y, dec):
    """dec 'global': one threshold (0.01 grid) for all labels; 'label': per-label thresholds (0.05 grid)."""
    if dec == "global":
        return np.full(S.shape[1], max(FINE, key=lambda t: _micro(Y, S >= t)))
    return fit_dec(S, Y, "label")


def nested_gen(oofs, Yt, Ft, out_mask, step, dec):
    """Weights + thresholds chosen on the tuning rows (Yt, Ft) of the other 4 folds; returns predictions on out_mask."""
    P = np.zeros(Yt.shape, bool)
    for k in range(5):
        tr, te = Ft != k, Ft == k; best = (-1, None, None)
        for w in weight_grid(len(oofs), step):
            S = sum(wi * o for wi, o in zip(w, oofs)); t = fit(S[tr], Yt[tr], dec); f = _micro(Yt[tr], S[tr] >= t)
            if f > best[0]:
                best = (f, w, t)
        _, w, t = best; P[te] = sum(wi * o[te] for wi, o in zip(w, oofs)) >= t
    return P[out_mask]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    cv = load_cv(); Y_all = np.stack(cv.labels).astype(bool); F_all = cv.fold.values
    en_mask = (cv.lang == "en").values
    en, Y, F = load_en()
    models = {}
    for n, r in TRAIN_CV.items():
        if (ROOT / "runs" / r / "oof.npz").exists():
            o, d, ch = load_train_cv(r, en_mask, F_all, Y_all, cv.lang.values)
            models[n] = (o, d, load_train_cv.full); print(f"{n:15s} epochs/fold {[c + 1 for c in ch]}  dev {'ok' if d is not None else 'MISSING'}")
    for n in DEBERTA:
        if (ROOT / "runs" / "step4" / n / "scores.npz").exists():
            o, d, ch = load_deberta(n, Y, F)
            if o is not None:
                models[n] = (o, d, None); print(f"{n:15s} epochs/fold {[c + 1 for c in ch]}  dev ok")
    rows = []
    names = list(models)
    for r in (1, 2, 3):
        for combo in itertools.combinations(names, r):
            step = 0.1 if r == 3 else 0.05
            tunes = ["en"] + (["all"] if all(models[c][2] is not None for c in combo) else [])
            for tag in tunes:
                for dec in (["global"] if r == 3 else ["global", "label"]):
                    if tag == "all":
                        P = nested_gen([models[c][2] for c in combo], Y_all, F_all, en_mask, step, dec)
                    else:
                        P = nested_gen([models[c][0] for c in combo], Y, F, np.ones(len(Y), bool), step, dec)
                    mi, ma = per_fold(Y, P, F)
                    rows.append(dict(system=" + ".join(combo) + f" [tune {tag}, thr {dec}]", micro=100 * mi.mean(),
                                     std=100 * mi.std(), macro=100 * ma.mean(), per_fold=(100 * mi).round(1).tolist(),
                                     combo=combo, tune=tag, dec=dec))
                print(f"{rows[-1]['system']:55s} micro {rows[-1]['micro']:.2f} ± {rows[-1]['std']:.2f}  macro {rows[-1]['macro']:.2f}", flush=True)
    rows.sort(key=lambda r: -r["micro"])
    best = rows[0]
    # stacker on top of the best subset
    from src.stack import features, nested as stack_nested
    X = features([models[c][0] for c in best["combo"]], en.article_id.values, en.para_idx.values)
    Ps, _ = stack_nested(X, Y, F, "global"); mi, ma = per_fold(Y, Ps, F)
    print(f"stacker over best ({best['system']}): micro {100 * mi.mean():.2f} ± {100 * mi.std():.2f}  macro {100 * ma.mean():.2f}")
    out = dict(table=[{k: v for k, v in r.items() if k != "combo"} for r in rows],
               stacker=dict(over=best["system"], micro=100 * mi.mean(), std=100 * mi.std(), macro=100 * ma.mean()))
    (ROOT / "runs" / "step10.json").write_text(json.dumps(out, indent=1))
    if not a.write:
        return
    # final: weights + thresholds on ALL EN OOF; only systems whose dev probabilities exist
    ok = [r for r in rows if all(models[c][1] is not None for c in r["combo"])]
    pick = ok[0]; combo = pick["combo"]
    step = 0.1 if len(combo) == 3 else 0.05
    oofs, Yt = ([models[c][2] for c in combo], Y_all) if pick["tune"] == "all" else ([models[c][0] for c in combo], Y)
    bw, bt, bf = None, None, -1
    for w in weight_grid(len(combo), step):
        S = sum(wi * o for wi, o in zip(w, oofs)); t = fit(S, Yt, pick["dec"]); f = _micro(Yt, S >= t)
        if f > bf:
            bw, bt, bf = w, t, f
    Sd = sum(wi * models[c][1] for wi, c in zip(bw, combo)); Pd = Sd >= bt
    dev = load_dev("en")
    path = ROOT / "runs" / "submissions" / "final_EN_dev.txt"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r, p in zip(dev.itertuples(), Pd):
            f.write("\t".join([r.doc, r.start, r.end] + [LABELS[i] for i in np.flatnonzero(p)]) + "\n")
    cfg = dict(system=pick["system"], tune=pick["tune"], dec=pick["dec"], weights=bw.round(2).tolist(), thresholds=bt.tolist(), cv_micro=pick["micro"],
               cv_std=pick["std"], cv_macro=pick["macro"])
    (ROOT / "configs").mkdir(exist_ok=True); (ROOT / "configs" / "final_en.json").write_text(json.dumps(cfg, indent=1))
    print(f"FINAL {pick['system']} w={cfg['weights']} -> {path}: {Pd.sum(1).mean():.2f} labels/para, "
          f"{100 * (Pd.sum(1) == 0).mean():.1f}% None")


if __name__ == "__main__":
    main()
