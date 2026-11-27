"""SR Phase 2 (run once): base vs model in use on HELDOUT (EN final split)."""
import json

import numpy as np
import torch

from sr.harness import SR, Runner, grade, metrics, pool_df

out = SR / "phase2.json"
if out.exists():
    raise SystemExit("Phase 2 already ran")
use = json.loads((SR / "model_in_use.json").read_text())
R = Runner(); df = pool_df("heldout", allow_heldout=True); Y = np.stack(df.labels).astype(bool)
Pb = R.predict(df); R.load(torch.load(use["path"])); Pe = R.predict(df); R.load(None)
mb, me = metrics(Y, Pb), metrics(Y, Pe); gb, ge = grade(Y, Pb), grade(Y, Pe)
arts = df.article_id.values; ua = np.unique(arts); rng = np.random.default_rng(0); d_mi, d_ma = [], []
for _ in range(5000):
    s = np.concatenate([np.flatnonzero(arts == a) for a in rng.choice(ua, len(ua))])
    a, b = metrics(Y[s], Pb[s]), metrics(Y[s], Pe[s]); d_mi.append(b["micro"] - a["micro"]); d_ma.append(b["macro"] - a["macro"])
r = dict(n=len(df), articles=len(ua), edit=use["edit"],
         micro_base=mb["micro"], micro_edit=me["micro"], macro_base=mb["macro"], macro_edit=me["macro"],
         d_micro_ci95=list(np.percentile(d_mi, [2.5, 97.5])), d_macro_ci95=list(np.percentile(d_ma, [2.5, 97.5])),
         wrong_to_right=int((~gb.right & ge.right).sum()), right_to_wrong=int((gb.right & ~ge.right).sum()),
         changed=int((Pb != Pe).any(1).sum()))
out.write_text(json.dumps(r, indent=1, default=float)); print(json.dumps(r, indent=1, default=float))
