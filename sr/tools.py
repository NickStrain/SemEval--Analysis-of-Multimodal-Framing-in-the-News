"""Editor-side helpers (analysis only; all model access goes through sr.harness actions).

  python -m sr.tools groups  <sid> <target>          -> sr/sessions/<sid>/read_ids.txt (20 fail + 20 ok)
  python -m sr.tools separate <sid> <readN>          -> per-layer separation of fail vs ok
  python -m sr.tools proj     <sid> <readN> <L> <a>  -> writes edits/sr/<sid>_projL<L>_a<a>.py
  python -m sr.tools bias     <sid> <c>              -> writes edits/sr/<sid>_bias<c>.py

Targets (fail vs ok, both from the base diagnosis on PRACTICE):
  F1  pred frame & gold None      vs  gold None & pred None
  F2  pred None & gold frames     vs  gold frames & exactly right
  F3  |pred| > |gold|+1           vs  exactly right (any gold)
  FP:<Label>  Label predicted & not gold   vs  Label neither predicted nor gold, paragraph exactly right
  FN:<Label>  Label gold & not predicted   vs  Label gold & predicted
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SR = Path(__file__).resolve().parent
EDITS = SR.parent / "edits" / "sr"


def diag():
    d = pd.read_csv(SR / "diagnosis.csv", keep_default_na=False)
    d["G"] = d.gold.str.split("|").apply(set)
    d["P"] = d.pred.str.split("|").apply(set)
    return d


def groups(target):
    d = diag()
    gn, pn = d.gold == "None", d.pred == "None"
    if target == "F1":
        fail, ok = d.F1.astype(str) == "True", gn & pn
    elif target == "F2":
        fail, ok = d.F2.astype(str) == "True", ~gn & (d.right.astype(str) == "True")
    elif target == "F3":
        fail, ok = d.F3.astype(str) == "True", d.right.astype(str) == "True"
    elif target.startswith("FP:"):
        lab = target[3:]
        inP, inG = d.P.apply(lambda s: lab in s), d.G.apply(lambda s: lab in s)
        fail, ok = inP & ~inG, ~inP & ~inG & (d.right.astype(str) == "True")
    elif target.startswith("FN:"):
        lab = target[3:]
        inP, inG = d.P.apply(lambda s: lab in s), d.G.apply(lambda s: lab in s)
        fail, ok = inG & ~inP, inG & inP
    else:
        raise ValueError(target)
    rng = np.random.default_rng(0)
    f = sorted(rng.choice(d.pid[fail].to_numpy(), min(20, fail.sum()), replace=False))
    o = sorted(rng.choice(d.pid[ok].to_numpy(), min(20, ok.sum()), replace=False))
    return f, o


def separate(sid, read):
    z = np.load(SR / "sessions" / sid / f"{read}.npz")
    g = json.loads((SR / "sessions" / sid / "groups.json").read_text())
    ids = list(z["ids"]); fi = [ids.index(i) for i in g["fail"]]; oi = [ids.index(i) for i in g["ok"]]
    res = []
    for k in [k for k in z.files if k.startswith("L")]:
        H = z[k]; u = H[fi].mean(0) - H[oi].mean(0); u /= np.linalg.norm(u)
        p = H @ u; pf, po = p[fi], p[oi]
        dprime = (pf.mean() - po.mean()) / np.sqrt((pf.var(ddof=1) + po.var(ddof=1)) / 2)
        # leave-one-out accuracy of a nearest-mean rule along u (less optimistic than d')
        acc = []
        for j in range(len(ids)):
            if j not in fi and j not in oi:
                continue
            F = [x for x in fi if x != j]; O = [x for x in oi if x != j]
            uu = H[F].mean(0) - H[O].mean(0); uu /= np.linalg.norm(uu)
            thr = ((H[F] @ uu).mean() + (H[O] @ uu).mean()) / 2
            acc.append(((H[j] @ uu) > thr) == (j in fi))
        res.append((int(k[1:]), dprime, np.mean(acc)))
    return res


PROJ = '''# Session {sid}: project u = unit(mean h_fail - mean h_ok) at hidden layer {L} out of
# encoder.layer.{li} attention-output and FFN-output weights, a = {a}.
import numpy as np

FAIL = {fail}
OK = {ok}


def edit(P, H):
    z = H.load_read("{read}")
    ids = list(z["ids"]); Hs = z["L{L}"]
    u = Hs[[ids.index(i) for i in FAIL]].mean(0) - Hs[[ids.index(i) for i in OK]].mean(0)
    u = H.unit(u)
    for name in ("enc.encoder.layer.{li}.attention.output.dense.weight", "enc.encoder.layer.{li}.output.dense.weight"):
        P[name] = H.project_out(P[name], u, {a})
'''

BIAS = '''# Session {sid}: shift intermediate.dense.bias (pre-GELU) in every layer by the constant {c}.
# (output.dense.bias / attention.output.dense.bias shifts are cancelled exactly by the following LayerNorm.)


def edit(P, H):
    for l in range(12):
        P[f"enc.encoder.layer.{{l}}.intermediate.dense.bias"] += {c}
'''

if __name__ == "__main__":
    cmd, sid = sys.argv[1], sys.argv[2]
    sdir = SR / "sessions" / sid; sdir.mkdir(parents=True, exist_ok=True); EDITS.mkdir(parents=True, exist_ok=True)
    if cmd == "groups":
        f, o = groups(sys.argv[3])
        (sdir / "groups.json").write_text(json.dumps(dict(target=sys.argv[3], fail=f, ok=o)))
        (sdir / "read_ids.txt").write_text("\n".join(f + o))
        print(f"{sys.argv[3]}: {len(f)} failing + {len(o)} ok paragraphs -> {sdir / 'read_ids.txt'}")
    elif cmd == "separate":
        for L, dp, acc in separate(sid, sys.argv[3]):
            print(f"layer {L:>2}: d' {dp:5.2f}   leave-one-out acc {acc:.2f}")
    elif cmd == "proj":
        read, L, a = sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
        g = json.loads((sdir / "groups.json").read_text())
        p = EDITS / f"{sid}_projL{L}_a{a:g}.py"
        p.write_text(PROJ.format(sid=sid, L=L, li=L - 1, a=a, read=read, fail=g["fail"], ok=g["ok"]))
        print(p)
    elif cmd == "bias":
        c = float(sys.argv[3]); p = EDITS / f"{sid}_bias{c:+g}.py"
        p.write_text(BIAS.format(sid=sid, c=c)); print(p)
