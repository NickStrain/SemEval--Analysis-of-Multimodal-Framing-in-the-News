"""Average mCPT variants into one member, then nested ensemble with XLM-R large + silver DeBERTa (EN)."""
import itertools, numpy as np
from src.en import per_fold
from src.score_mcpt import Y, F, en_oof, nested_epoch, nested_ens, xl, db
R = {"orig": "phase1b/M_E1_extraEN", "S1": "phase2m/S1_silver", "S3": "phase2m/S3_trans_enonly",
     "C1": "phase2m/C1_joint", "C2": "phase2m/C2_pre_then_bce", "C3": "phase2m/C3_pre_joint"}
O = {k: nested_epoch(en_oof(v))[0] for k, v in R.items()}
res = []
for r in range(1, 7):
    for c in itertools.combinations(O, r):
        m = np.mean([O[x] for x in c], 0)
        a, _ = per_fold(Y, nested_ens([m]), F); e, _ = per_fold(Y, nested_ens([m, xl, db]), F)
        res.append((100 * e.mean(), 100 * e.std(), 100 * a.mean(), 100 * a.std(), "+".join(c)))
for e, es, a, as_, n in sorted(res, reverse=True)[:8]:
    print(f"mCPT avg [{n:22s}] alone {a:.2f} ± {as_:.2f} | ensemble {e:.2f} ± {es:.2f}")
print("best mCPT-avg alone:", max(res, key=lambda x: x[2])[4], round(max(r[2] for r in res), 2))
