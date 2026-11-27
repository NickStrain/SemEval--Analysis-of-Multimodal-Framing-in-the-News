"""Step 2: post-processing search for a model's OOF probabilities (cross-fitted), vs a baseline per-fold file."""
import json
import sys

import numpy as np

from src.en import keep, load_en, log_run, per_fold, summary
from src.post import cv_apply, fit

name = sys.argv[1]
base_file = sys.argv[2] if len(sys.argv) > 2 else "runs/step1_mcpt_per_fold.npy"
en, Y, F = load_en(); S = np.load(f"oof/{name}.npy")
P, prms = cv_apply(S, Y, F)
mi, ma = per_fold(Y, P, F); base = np.load(base_file)
print(f"{name} + post (cross-fitted): {summary(mi, ma)}")
print("  per fold micro:", " ".join(f"{100 * x:.1f}" for x in mi))
print("  per-fold choices (m, k):", [(p["m"], p["k"]) for p in prms])
ok, msg = keep(mi, base[:, 0]); print("  vs baseline:", msg)
full = fit(S, Y)
cfg = "configs/post_mcpt.json" if name == "mcpt_en" else f"configs/post_{name}.json"
json.dump(full, open(cfg, "w"), indent=1)
np.save(f"runs/step2_{name}_per_fold.npy", np.c_[mi, ma]); np.save(f"oof/{name}_postpred.npy", P)
log_run(2, name, f"post m={full['m']} k={full['k']}", mi, ma)
print(f"  full-OOF fit saved to {cfg}:", {"m": full["m"], "k": full["k"]})
