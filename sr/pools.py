"""Phase 0.1 — English pools for the self-rewriting harness (seed 0, split by article, frozen by hash).

  RESERVED : 15% of the English CV articles. Never visible to editing sessions.
  PRACTICE : English CV articles drawn until >= 300 paragraphs (diagnosis uses 256 of them).
  BASE-TRAIN: the remaining English CV articles. The base model (M_E1 recipe) is retrained on these
             + all other languages + batch-1 English, so PRACTICE/RESERVED are unseen text.
  HELDOUT  : English articles of the locked final split (+ one Codabench dev upload in Phase 2).

Run once: python -m sr.pools  -> sr/pools.json (+ sha256 inside). Refuses to overwrite.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from src.data import SPLITS_FILE, load_cv

OUT = Path(__file__).resolve().parent / "pools.json"


def digest(p: dict) -> str:
    body = {k: p[k] for k in ("seed", "practice", "reserved", "base_train", "heldout_final")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def load_pools() -> dict:
    p = json.loads(OUT.read_text())
    if digest(p) != p["sha256"]:
        raise RuntimeError("sr/pools.json does not match its hash: pools were modified")
    return p


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"{OUT} exists and is frozen (sha256 {load_pools()['sha256'][:12]}...)")
    cv = load_cv(); en = cv[cv.lang == "en"]
    n_par = en.groupby("article_id").size()
    arts = list(np.random.default_rng(0).permutation(sorted(n_par.index)))
    n_res = round(0.15 * len(arts))
    reserved, rest = arts[:n_res], arts[n_res:]
    practice, i = [], 0
    while n_par[practice].sum() < 300 if practice else True:
        practice.append(rest[i]); i += 1
    base_train = rest[i:]
    final = json.loads(SPLITS_FILE.read_text())["final"]
    heldout = sorted(a for a in final if a.startswith("EN_"))
    p = dict(seed=0, practice=sorted(map(str, practice)), reserved=sorted(map(str, reserved)),
             base_train=sorted(map(str, base_train)), heldout_final=heldout, heldout_dev="Codabench EN dev (1 upload)")
    p["sha256"] = digest(p)
    OUT.write_text(json.dumps(p, indent=1))
    for k in ("practice", "reserved", "base_train"):
        print(f"{k:<10} {len(p[k]):>3} articles {int(n_par[p[k]].sum()):>4} paragraphs")
    print(f"heldout    {len(heldout):>3} EN final articles (locked until Phase 2)\nsha256 {p['sha256']}")
