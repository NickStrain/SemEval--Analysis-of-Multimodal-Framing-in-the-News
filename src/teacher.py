"""LLM-teacher pilot (Step 7): Claude labels English paragraphs WITHOUT seeing gold; then scored officially.

  python -m src.teacher pilot   -> teacher/input/pilot_XX.md  (articles: title + numbered paragraphs, no labels)
  python -m src.teacher score   -> teacher micro/macro vs gold, and mCPT (M_E1 OOF, cross-fitted) on the same paragraphs
"""
import json
import sys
from pathlib import Path

import numpy as np

from src.data import LABEL2ID, LABELS, ROOT, load_cv
from src.decode import crossfit
from src.eval import official_scores

T = ROOT / "teacher"
N_ARTICLES = 20


def pilot():
    cv = load_cv(); en = cv[cv.lang == "en"]
    arts = sorted(en.article_id.unique())
    pick = sorted(np.random.default_rng(0).choice(arts, N_ARTICLES, replace=False))
    (T / "pilot_articles.json").write_text(json.dumps(pick))
    chunk, n, k = [], 0, 0
    for a in pick:
        g = en[en.article_id == a].sort_values("para_idx")
        lines = [f"## ARTICLE {a}", f"TITLE: {g.title.iloc[0]}", ""]
        for r in g.itertuples():
            lines.append(f"[{a}#{r.para_idx}] {r.text}\n")
        chunk.append("\n".join(lines)); n += len(g)
        if n >= 50:
            (T / "input" / f"pilot_{k:02d}.md").write_text("\n".join(chunk)); chunk, n, k = [], 0, k + 1
    if chunk:
        (T / "input" / f"pilot_{k:02d}.md").write_text("\n".join(chunk)); k += 1
    print(f"{N_ARTICLES} articles, {int(en.article_id.isin(pick).sum())} paragraphs, {k} chunk files")


def load_teacher():
    out = {}
    for f in sorted((T / "labels").glob("pilot_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line); out[r["pid"]] = r
    return out


def score():
    cv = load_cv(); cv["pid"] = cv.article_id + "#" + cv.para_idx.astype(str)
    lab = load_teacher()
    m = cv.pid.isin(lab.keys()).values
    Y = np.stack(cv.labels).astype(bool)
    P = np.zeros_like(Y)
    for i in np.flatnonzero(m):
        for f, p in lab[cv.pid[i]]["p"].items():
            if p >= 0.5:
                P[i, LABEL2ID[f]] = True
    # mCPT M_E1 out-of-fold predictions (cross-fitted thresholds, implicit None) on the same rows
    S = np.load(ROOT / "runs/phase1b/M_E1_extraEN/oof.npz")["scores"]
    Pm, _ = crossfit(S, Y, cv.fold.values, cv.lang.values, "implicit", False)
    ti, ta = official_scores(Y[m], P[m]); mi, ma = official_scores(Y[m], Pm[m])
    rng = np.random.default_rng(0); arts = cv.article_id[m].unique(); diffs = []
    for _ in range(2000):  # article-level bootstrap of the teacher - mCPT micro difference
        s = np.isin(cv.article_id.values, rng.choice(arts, len(arts))) & m
        diffs.append(official_scores(Y[s], P[s])[0] - official_scores(Y[s], Pm[s])[0])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"paragraphs scored: {m.sum()}  (missing labels: {len(set(cv.pid[cv.lang == 'en']) & set(lab)) - m.sum()})")
    print(f"TEACHER (Claude)   micro {100 * ti:.1f}  macro {100 * ta:.1f}")
    print(f"mCPT M_E1 (OOF)    micro {100 * mi:.1f}  macro {100 * ma:.1f}")
    print(f"teacher - mCPT micro: {100 * (ti - mi):+.1f}  95% CI [{100 * lo:+.1f}, {100 * hi:+.1f}] (article bootstrap)")
    t15 = np.c_[Y[m], ~Y[m].any(1)]; p15 = np.c_[P[m], ~P[m].any(1)]
    print("\nper label (teacher)   TP  FP  FN")
    for j, l in enumerate(LABELS + ["None"]):
        print(f"  {l:<46}{(t15[:, j] & p15[:, j]).sum():>4}{(~t15[:, j] & p15[:, j]).sum():>4}{(t15[:, j] & ~p15[:, j]).sum():>4}")


if __name__ == "__main__":
    {"pilot": pilot, "score": score}[sys.argv[1]]()
