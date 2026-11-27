"""Check src.eval against the official scorer on 3 synthetic prediction files (non-final data only)."""
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.data import LABELS, ROOT, load_cv
from src.eval import official_scores, write_official

OUT = ROOT / "runs" / "verify_eval"
OUT.mkdir(parents=True, exist_ok=True)
SCORER = ROOT / "starter-kit/starter-kit-v2/subtask1/scorer-subtask-1.py"

df = load_cv()
keys = [(a + ".txt", s, e) for a, s, e in zip(df.article_id, df.start, df.end)]
Y = np.stack(df.labels)
rng = np.random.default_rng(0)
(OUT / "frames.txt").write_text("\n".join(LABELS) + "\n")
write_official(OUT / "gold.tsv", keys, Y)

flip = rng.random(Y.shape) < 0.08
synthetic = {
    "noisy_gold": np.where(flip, 1 - Y, Y),                       # 8% of bits flipped
    "random": (rng.random(Y.shape) < 0.12).astype(np.int8),       # many empty rows -> None
    "all_empty": np.zeros_like(Y),                                # predicts None everywhere
}
ok = True
for name, P in synthetic.items():
    write_official(OUT / f"{name}.tsv", keys, P)
    r = subprocess.run([sys.executable, str(SCORER), "-g", OUT / "gold.tsv", "-p", OUT / f"{name}.tsv",
                        "-f", OUT / "frames.txt", "-o"], capture_output=True, text=True, check=True)
    off_mi, off_ma = map(float, r.stdout.strip().split("\t"))
    mi, ma = official_scores(Y, P)
    same = abs(mi - off_mi) < 1e-12 and abs(ma - off_ma) < 1e-12
    ok &= same
    print(f"{name:<11} official micro {off_mi:.6f} macro {off_ma:.6f} | ours micro {mi:.6f} macro {ma:.6f} "
          f"| {'IDENTICAL' if same else 'MISMATCH'}")
sys.exit(0 if ok else 1)
