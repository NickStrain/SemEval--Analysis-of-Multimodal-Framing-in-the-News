"""Average several mCPT variants (each: epoch nested per fold), then one nested global threshold. EN only."""
import itertools, numpy as np
from src.en import per_fold
from src.score_mcpt import Y, F, en_oof, nested_epoch, nested_ens, xl, db
runs = {"orig": "phase1b/M_E1_extraEN", "silver": "phase2m/S1_silver", "trans_en": "phase2m/S3_trans_enonly"}
O = {k: nested_epoch(en_oof(v))[0] for k, v in runs.items()}
# also: average of the last 4 epochs (smoother than picking one epoch)
for k, v in runs.items():
    z = en_oof(v); O[k + "_avgL4"] = z[-4:].mean(0)
for r in (1, 2, 3):
    for c in itertools.combinations(O, r):
        if len({x.split("_avg")[0] for x in c}) < r: continue
        mi, ma = per_fold(Y, nested_ens([O[x] for x in c]), F)
        print(f"{' + '.join(c):40s} {100*mi.mean():.2f} ± {100*mi.std():.2f}  macro {100*ma.mean():.1f}", flush=True)
