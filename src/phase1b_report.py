"""Phase 1b: does extra / reweighted English data help? Compared to E0 fold by fold.

Usage: python -m src.phase1b_report
Writes runs/phase1b/report.md and runs/phase1b/results.json.

Rule (English-focused, same shape as the edit gate): keep a change iff EN micro-F1 improves by
>= 0.5 pt on average AND in >= 4 of 5 folds; other languages are reported so regressions are visible.

E4 (English-only stage 2) is an English specialist: it is scored on EN rows only, with thresholds and
checkpoint cross-fitted on the other folds' EN rows. Its reference is E3 / E0 scored the same way.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.data import LANGS, ROOT, load_cv
from src.decode import crossfit
from src.eval import official_scores

df = load_cv()
Y = np.stack(df.labels).astype(bool)
FOLDS, LANG = df.fold.values, df.lang.values
EN = LANG == "en"
R = ROOT / "runs"

BACKBONES = {
    "mCPT (ctx, lr 5e-5)": ("M", R / "phase1" / "B_ctx_lr5e-5"),
    "XLM-R base (lr 2e-5)": ("X", R / "phase1" / "C_xlmr_base"),
}
VARIANTS = [("E1 +batch-1 EN", "E1_extraEN"), ("E2 EN x2", "E2_over2"),
            ("E3 +batch-1 EN, EN x2", "E3_extraEN_over2"), ("E5 +batch-1 EN+RU", "E5_extraENRU")]
MODE = "implicit"  # best None handling for both backbones in Phase 1


def all_langs(path):
    z = np.load(path)
    pred, chosen = crossfit(z["scores"], Y, FOLDS, LANG, MODE, False)
    out = []
    for k in range(5):
        m = FOLDS == k
        rec = dict(zip(("micro", "macro"), official_scores(Y[m], pred[m])))
        for lg in LANGS:
            rec[lg] = official_scores(Y[m & (LANG == lg)], pred[m & (LANG == lg)])[0]
        out.append(rec)
    return out


def en_only(path):
    z = np.load(path)
    s = z["scores"][:, EN]
    pred, _ = crossfit(s, Y[EN], FOLDS[EN], LANG[EN], MODE, False)
    return [dict(zip(("en_micro", "en_macro"), official_scores(Y[EN][FOLDS[EN] == k], pred[FOLDS[EN] == k])))
            for k in range(5)]


def job(item):
    key, kind, path = item
    return key, (all_langs(path) if kind == "all" else en_only(path))


def mean_std(vals):
    return f"{100 * np.mean(vals):.1f} ± {100 * np.std(vals):.1f}"


def diff(a, b, key):
    d = [x[key] - y[key] for x, y in zip(a, b)]
    m, n = 100 * np.mean(d), sum(v > 0 for v in d)
    ok = m >= 0.5 and n >= 4
    return f"{m:+.1f} ({n}/5){' ✅' if ok else ''}"


if __name__ == "__main__":
    jobs = []
    for bname, (p, base) in BACKBONES.items():
        jobs += [((bname, "E0"), "all", base / "oof.npz"), ((bname, "E0", "en"), "en", base / "oof.npz")]
        for vname, suffix in VARIANTS:
            f = R / "phase1b" / f"{p}_{suffix}" / "oof.npz"
            if f.exists():
                jobs.append(((bname, vname), "all", f))
                if suffix.startswith("E3"):
                    jobs.append(((bname, vname, "en"), "en", f))
                    f2 = f.parent / "oof_stage2.npz"
                    if f2.exists():
                        jobs.append(((bname, "E4 = E3 + EN-only stage 2", "en"), "en", f2))
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = dict(ex.map(job, jobs))

    L = ["Official F1 ×100, mean ± std over 5 folds (None: implicit, global thresholds, cross-fitted).",
         "Δ = mean change vs E0 (folds better /5); ✅ = passes the rule (≥ +0.5 and ≥ 4/5 folds).", ""]
    for bname in BACKBONES:
        base = res.get((bname, "E0"))
        L += [f"### {bname}", "",
              "| run | micro | macro | **EN** | BG | PT | RU | Δ EN | Δ micro | Δ macro |", "|---|---|---|---|---|---|---|---|---|---|"]
        for vname in ["E0"] + [v for v, _ in VARIANTS]:
            r = res.get((bname, vname))
            if r is None:
                continue
            cells = [mean_std([x[c] for x in r]) for c in ("micro", "macro", "en", "bg", "pt", "ru")]
            ds = ["–"] * 3 if vname == "E0" else [diff(r, base, c) for c in ("en", "micro", "macro")]
            L.append(f"| {vname} | " + " | ".join(cells) + " | " + " | ".join(ds) + " |")
        L += ["", "English specialist (EN rows only, thresholds tuned on EN only):", "",
              "| run | EN micro | EN macro | Δ EN micro vs E0 | Δ vs E3 |", "|---|---|---|---|---|"]
        e0 = res.get((bname, "E0", "en"))
        e3 = res.get((bname, "E3 +batch-1 EN, EN x2", "en"))
        for vname in ["E0", "E3 +batch-1 EN, EN x2", "E4 = E3 + EN-only stage 2"]:
            r = res.get((bname, vname, "en"))
            if r is None:
                continue
            d0 = "–" if vname == "E0" else diff(r, e0, "en_micro")
            d3 = diff(r, e3, "en_micro") if vname.startswith("E4") and e3 else "–"
            L.append(f"| {vname} | {mean_std([x['en_micro'] for x in r])} | {mean_std([x['en_macro'] for x in r])} | {d0} | {d3} |")
        L.append("")
    (R / "phase1b" / "report.md").write_text("\n".join(L))
    (R / "phase1b" / "results.json").write_text(json.dumps({" / ".join(k): v for k, v in res.items()}, indent=1))
    print("\n".join(L))
