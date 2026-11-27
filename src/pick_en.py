"""Print the CV run with the best English micro-F1 (None=implicit, global thresholds, cross-fitted)."""
import sys

import numpy as np

from src.data import ROOT, load_cv
from src.decode import crossfit
from src.eval import official_scores

df = load_cv(); Y = np.stack(df.labels).astype(bool); F, L = df.fold.values, df.lang.values; EN = L == "en"
best = None
for run in sys.argv[1:]:
    p = ROOT / "runs" / run / "oof.npz"
    if not p.exists():
        print(f"# {run}: missing", file=sys.stderr); continue
    pred, _ = crossfit(np.load(p)["scores"], Y, F, L, "implicit", False)
    en = np.mean([official_scores(Y[EN & (F == k)], pred[EN & (F == k)])[0] for k in range(5)])
    print(f"# {run}: EN micro {100 * en:.1f}", file=sys.stderr)
    if best is None or en > best[0]:
        best = (en, run)
print(best[1])
