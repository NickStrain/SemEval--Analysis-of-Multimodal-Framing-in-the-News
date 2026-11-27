"""Data loading for SemEval-2027 Task 9, Subtask 1 (paragraph-level framing).

Format (per language dir): subtask-1-annotations-with-paragraphs.txt, TSV:
    doc_id  start  end  label_1 ... label_k  paragraph_text
A paragraph with no frame has the single placeholder label "-".
The title is the first line of raw-documents/<doc_id>.

One row per paragraph:
    article_id, lang, para_idx, title, text, prev_text, next_text, labels (14-dim multi-hot)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPLITS_FILE = DATA / "splits.json"

# Canonical order (README_ST1.md).
LABELS = [
    "Economic",
    "Capacity_and_resources",
    "Morality",
    "Fairness_and_equality",
    "Legality_Constitutionality_and_jurisprudence",
    "Policy_prescription_and_evaluation",
    "Crime_and_punishment",
    "Security_and_defense",
    "Health_and_safety",
    "Quality_of_life",
    "Cultural_identity",
    "Public_opinion",
    "Political",
    "External_regulation_and_reputation",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
LANGS = ["bg", "en", "pt", "ru"]

# Labelled Task-9 sources. 3_dev-set-nov-release is unlabelled (participant files) -> not used here.
SOURCES = {
    "bg": DATA / "2_training-second-batch-oct-revised" / "BG",
    "en": DATA / "2_training-second-batch-oct-revised" / "EN",
    "pt": DATA / "2_training-second-batch-oct-revised" / "PT",
    "ru": DATA / "4_training-russian-nov" / "RU",
}


def _read_lang(lang: str, src: Path) -> list[dict]:
    rows = []
    raw_dir = src / "raw-documents"
    titles: dict[str, str] = {}
    with open(src / "subtask-1-annotations-with-paragraphs.txt", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            doc, start, end = parts[0], int(parts[1]), int(parts[2])
            labels, text = parts[3:-1], parts[-1]
            labels = [l for l in labels if l and l != "-"]
            unknown = set(labels) - LABEL2ID.keys()
            if unknown:
                raise ValueError(f"{src}:{ln}: unknown labels {unknown}")
            if doc not in titles:
                raw = (raw_dir / doc).read_text(encoding="utf-8")
                titles[doc] = raw.split("\n", 1)[0].strip()
            rows.append(dict(article_id=doc.removesuffix(".txt"), lang=lang, start=start, end=end,
                             title=titles[doc], text=text, label_names=labels))
    return rows


def load_all() -> pd.DataFrame:
    """All labelled paragraphs, every split. Internal: use load_split() for experiments."""
    rows = []
    for lang, src in SOURCES.items():
        rows += _read_lang(lang, src)
    df = pd.DataFrame(rows).sort_values(["lang", "article_id", "start"], kind="stable").reset_index(drop=True)
    df["para_idx"] = df.groupby("article_id").cumcount()
    g = df.groupby("article_id")["text"]
    df["prev_text"] = g.shift(1).fillna("")
    df["next_text"] = g.shift(-1).fillna("")
    mh = np.zeros((len(df), len(LABELS)), dtype=np.int8)
    for i, names in enumerate(df["label_names"]):
        for n in names:
            mh[i, LABEL2ID[n]] = 1
    df["labels"] = list(mh)
    return df[["article_id", "lang", "para_idx", "title", "text", "prev_text", "next_text",
               "labels", "label_names", "start", "end"]]


# ---------------- extra training data: batch 1 (SemEval-2023 articles, paragraph-level) ----------------
# Only FRAMED paragraphs are listed (no None rows; frameless paragraphs are simply absent), so these
# rows must never train the None output. Training-only: never used for evaluation.
B1 = DATA / "1_initial-training-data" / "subtask1"
EXTRA_SOURCES = {
    "en": [B1 / "SEMEVAL23_FRAMING_INITIAL_TRAIN-B" / "EN"],
    "ru": [B1 / "SEMEVAL23_FRAMING_INITIAL_TRAIN-A" / "RU", B1 / "SEMEVAL23_FRAMING_INITIAL_TRAIN-B" / "RU"],
}


def _norm(s: str) -> str:
    return " ".join(s.split())


def _load_2b_gold(lang: str) -> pd.DataFrame:
    """Gold ST1 labels released with the dev set for the Subtask-2b articles (different articles from the ST1 dev set;
    checked: no overlap with ST1 dev or training). Includes real None paragraphs. Text by raw-block order."""
    import re
    d = DATA / "3_dev-set-nov-release" / lang.upper()
    f = d / "subtask-1-annotations-for-subtask-2b-participant.txt"
    if not f.exists():
        return pd.DataFrame()
    rows = [[x.strip() for x in l.rstrip("\r\n").split("\t")] for l in open(f, encoding="utf-8-sig") if l.strip()]
    out = []
    for doc in dict.fromkeys(r[0] for r in rows):
        rs = sorted([r for r in rows if r[0] == doc], key=lambda r: int(r[1]))
        raw = (d / "raw-documents" / doc).read_text(encoding="utf-8").replace("\r\n", "\n")
        blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
        if len(blocks) != len(rs):
            continue
        for i, r in enumerate(rs):
            labs = [l for l in r[3:] if l]
            y = np.zeros(len(LABELS), np.int8)
            for l in labs:
                y[LABEL2ID[l]] = 1
            out.append(dict(article_id=f"D2B_{doc.removesuffix('.txt')}", lang=lang, title=blocks[0], text=blocks[i],
                            prev_text=blocks[i - 1] if i else "", next_text=blocks[i + 1] if i + 1 < len(blocks) else "",
                            labels=y, label_names=labs, mask_none=False))
    return pd.DataFrame(out)


def load_extra(langs=("en",)) -> pd.DataFrame:
    import re
    rows = []
    for lang in langs:
        for src in EXTRA_SOURCES.get(lang, []):
            tag = src.parent.name[-1]  # A / B -> keeps ids unique across the two folders
            raws = {}
            with open(src / "subtask-1-annotations-with-paragraphs.txt", encoding="utf-8-sig") as f:
                for line in f:
                    parts = [p.strip() for p in line.rstrip("\r\n").split("\t")]
                    if len(parts) < 4 or not parts[0]:
                        continue
                    doc, labels, text = parts[0], [l for l in parts[3:-1] if l and l != "-"], parts[-1]
                    unknown = set(labels) - LABEL2ID.keys()
                    if unknown:
                        raise ValueError(f"{src}: unknown labels {unknown}")
                    if not labels:
                        continue
                    if doc not in raws:
                        raws[doc] = (src / "raw-documents" / doc).read_text(encoding="utf-8").replace("\r\n", "\n")
                    raw = raws[doc]
                    title = raw.split("\n", 1)[0].strip()
                    # context from the raw document (unlisted frameless paragraphs still count as neighbours)
                    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
                    nb = [_norm(b) for b in blocks]
                    nt = _norm(text)
                    idx = [i for i, b in enumerate(nb) if nt[:80] and nt[:80] in b]
                    prev_t = next_t = ""
                    if idx:
                        i0 = idx[0]
                        j1 = max(i for i in range(i0, len(nb)) if i == i0 or nb[i][:40] in nt) if nt else i0
                        prev_t = blocks[i0 - 1] if i0 >= 1 and blocks[i0 - 1] != title else ""
                        next_t = blocks[j1 + 1] if j1 + 1 < len(blocks) else ""
                    y = np.zeros(len(LABELS), np.int8)
                    for l in labels:
                        y[LABEL2ID[l]] = 1
                    rows.append(dict(article_id=f"B1{tag}_{doc.removesuffix('.txt')}", lang=lang, title=title,
                                     text=text, prev_text=prev_t, next_text=next_t, labels=y, label_names=labels,
                                     mask_none=True))
    # + gold ST1 labels of the dev-release Subtask-2b articles (added 2026-11-26 overnight)
    return pd.concat([pd.DataFrame(rows)] + [_load_2b_gold(l) for l in langs], ignore_index=True)


N_FOLDS = 5


def make_splits(df: pd.DataFrame, final_ids: list[str] | None = None, seed: int = 13,
                final_frac: float = 0.1, n_folds: int = N_FOLDS) -> dict:
    """Article-level, language-stratified: a locked 'final' split + n_folds CV folds over the rest.

    If final_ids is given (re-split), the final split is kept exactly as is.
    """
    rng = np.random.default_rng(seed)
    final, folds = [], [[] for _ in range(n_folds)]
    for lang in LANGS:
        arts = sorted(df.loc[df.lang == lang, "article_id"].unique())
        if final_ids is None:
            arts = list(rng.permutation(arts))
            n_fin = max(1, round(len(arts) * final_frac))
            final += arts[:n_fin]
            rest = arts[n_fin:]
        else:
            rest = [a for a in arts if a not in set(final_ids)]
        rest = list(rng.permutation(rest))
        offset = len(folds[0]) % n_folds  # rotate so small languages don't always fill fold 0 first
        for i, a in enumerate(rest):
            folds[(i + offset) % n_folds].append(str(a))
    final = sorted(final_ids if final_ids is not None else final)
    return {"seed": seed, "final": final, "folds": [sorted(f) for f in folds]}


def load_cv() -> pd.DataFrame:
    """All non-final paragraphs with a 'fold' column (0..N_FOLDS-1). Never contains final articles."""
    sp = json.loads(SPLITS_FILE.read_text())
    fold_of = {a: k for k, ids in enumerate(sp["folds"]) for a in ids}
    df = load_all()
    df = df[~df.article_id.isin(set(sp["final"]))].reset_index(drop=True)
    assert df.article_id.isin(fold_of.keys()).all()
    df["fold"] = df.article_id.map(fold_of)
    return df


def load_final(*, allow_final: bool = False) -> pd.DataFrame:
    """The LOCKED final split. Phase 4 only."""
    if not allow_final:
        raise PermissionError("The 'final' split is locked until Phase 4.")
    ids = set(json.loads(SPLITS_FILE.read_text())["final"])
    df = load_all()
    return df[df.article_id.isin(ids)].reset_index(drop=True)


if __name__ == "__main__":
    import sys
    df = load_all()
    old = json.loads(SPLITS_FILE.read_text()) if SPLITS_FILE.exists() else None
    if old and "folds" in old and "--force" not in sys.argv:
        print(f"{SPLITS_FILE} already has CV folds; not overwriting (use --force).")
        sys.exit()
    # Keep the locked final split from Phase 0 (old format stored it under splits.final).
    final_ids = old["splits"]["final"] if old and "splits" in old else (old or {}).get("final")
    sp = make_splits(df, final_ids=final_ids)
    SPLITS_FILE.write_text(json.dumps(sp, indent=1))
    print(f"wrote {SPLITS_FILE}: final={len(sp['final'])}, folds={[len(f) for f in sp['folds']]}")
