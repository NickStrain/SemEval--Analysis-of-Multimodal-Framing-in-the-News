"""Self-rewriting harness for English Subtask 1 (protocol as specified by the user, Phase 0).

Base model : ckpt/sr_base_M_E1.pt — the M_E1 recipe (mCPT + context + frame head, batch-1 EN) retrained
             WITHOUT the PRACTICE/RESERVED articles; decoding = its frozen thresholds, None = implicit.
Pools      : sr/pools.json (seed 0, by article, sha256-checked on every load).
Metric     : official micro-F1, 15 labels (14 frames + None).

Per paragraph:  s = TP - FP - FN over the 15 labels.
Failure types:  F1 = predicted a frame, gold is None
                F2 = predicted None, gold has frames
                F3 = |pred| > |gold| + 1            (sizes of the 15-label sets)

Actions (one per turn, each with a rationale of <= 2 sentences), logged in sr/sessions/<sid>/log.jsonl:
  code     --file F.py                      apply an edit (budget 200 per session)
  generate --ids SPEC --which base|edit|both [--edit NAME]   predict on PRACTICE ids  } shared budget
  read     --ids SPEC --layers L,..  [--edit NAME]           pooled hidden states     } of 20 calls
  submit   --edit NAME                      ONE paired test on RESERVED (once per session)
  decide   --decision accept|withdraw       accept only if reserved micro-F1 went up

An edit file defines  edit(P, H)  and changes tensors in P in place (all under no_grad):
  P : dict name -> tensor copy of the EDITABLE parameters only
  H : helpers  project_out(W, u, a) = W - a u u^T W,  unit(v),  load_read(name) -> dict,  diagnosis()
Rules enforced: same shapes; embeddings, final LayerNorm and head are frozen; no whole-tensor rescaling.

Usage:  python -m sr.harness diagnose
        python -m sr.harness <sid> <action> --why "..." [options]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from src.data import LABELS, ROOT, load_all, load_cv
from src.decode import decode, with_none
from src.train_cv import DEV, MODELS, Clf, encode_inputs, pad
from sr.pools import load_pools

SR = Path(__file__).resolve().parent
CKPT = ROOT / "ckpt" / "sr_base_M_E1.pt"
CODE_BUDGET, GEN_READ_BUDGET = 200, 20
N_DIAG = 256
LAST = 11  # index of the last transformer layer: its output.LayerNorm is the "final LayerNorm"


# ---------------------------------------------------------------- data
def _key(df):
    return (df.article_id + "#" + df.para_idx.astype(str)).tolist()


def pool_df(name: str, allow_heldout: bool = False) -> pd.DataFrame:
    p = load_pools()
    if name in ("practice", "reserved"):
        cv = load_cv()
        df = cv[cv.article_id.isin(set(p[name]))]
    elif name == "heldout":
        if not allow_heldout:
            raise PermissionError("HELDOUT is used only in Phase 2")
        a = load_all()
        df = a[a.article_id.isin(set(p["heldout_final"]))]
    else:
        raise ValueError(name)
    df = df.reset_index(drop=True)
    df["pid"] = _key(df)
    return df


def diag_ids() -> list[str]:
    ids = pool_df("practice").pid.tolist()
    return sorted(np.random.default_rng(0).choice(ids, N_DIAG, replace=False).tolist())


# ---------------------------------------------------------------- model
class Runner:
    def __init__(self):
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        self.cfg, self.T, self.mode = ck["config"], np.array(ck["thresholds"]), ck["mode"]
        enc, tok_name, pooling = MODELS[self.cfg["model"]]
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tok_name)
        self.model = Clf(enc, pooling, self.cfg["n_out"], self.cfg["ctx"]).to(DEV).eval()
        self.model.load_state_dict(ck["state_dict"])
        self.base_sd = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    @staticmethod
    def frozen(name: str) -> bool:
        return (name.startswith("enc.embeddings.") or name.startswith("head.") or name.startswith("enc.pooler.")
                or name.startswith(f"enc.encoder.layer.{LAST}.output.LayerNorm."))

    def editable(self) -> dict:
        return {k: v.clone() for k, v in self.base_sd.items() if not self.frozen(k)}

    def load(self, changed: dict | None):
        sd = dict(self.base_sd)
        if changed:
            sd.update({k: v.to(DEV) for k, v in changed.items()})
        self.model.load_state_dict(sd)

    @torch.no_grad()
    def scores(self, df) -> np.ndarray:
        tgt, ctx = encode_inputs(self.tok, df)
        out = []
        for b in range(0, len(df), 64):
            idx = range(b, min(b + 64, len(df)))
            ids, mask = pad([tgt[i] for i in idx], self.tok.pad_token_id)
            kw = dict(zip(("cids", "cmask"), pad([ctx[i] for i in idx], self.tok.pad_token_id))) if self.cfg["ctx"] else {}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(torch.sigmoid(self.model(ids, mask, **kw).float()).cpu().numpy())
        S = np.full((len(df), 15), np.nan, np.float32)
        S[:, : self.cfg["n_out"]] = np.concatenate(out)
        return S

    def predict(self, df) -> np.ndarray:
        return decode(self.scores(df), self.T, self.mode)

    @torch.no_grad()
    def hidden(self, df, layers) -> dict:
        tgt, _ = encode_inputs(self.tok, df)
        H = {l: [] for l in layers}
        for b in range(0, len(df), 64):
            ids, mask = pad([tgt[i] for i in range(b, min(b + 64, len(df)))], self.tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hs = self.model.enc(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
            m = mask.unsqueeze(-1).float()
            for l in layers:
                H[l].append(((hs[l].float() * m).sum(1) / m.sum(1)).cpu().numpy())
        return {l: np.concatenate(v) for l, v in H.items()}


# ---------------------------------------------------------------- grading
def grade(gold14, pred14) -> pd.DataFrame:
    t, p = with_none(gold14), with_none(pred14)
    tp, fp, fn = (t & p).sum(1), (~t & p).sum(1), (t & ~p).sum(1)
    return pd.DataFrame(dict(s=tp - fp - fn, tp=tp, fp=fp, fn=fn,
                             F1=t[:, 14] & p[:, :14].any(1), F2=p[:, 14] & t[:, :14].any(1),
                             F3=p.sum(1) > t.sum(1) + 1, right=(t == p).all(1)))


def metrics(gold14, pred14) -> dict:
    t, p = with_none(gold14), with_none(pred14)
    return dict(micro=f1_score(t, p, average="micro", zero_division=1),
                macro=f1_score(t, p, average="macro", zero_division=1))


def per_label(gold14, pred14) -> pd.DataFrame:
    t, p = with_none(gold14), with_none(pred14)
    return pd.DataFrame(dict(TP=(t & p).sum(0), FP=(~t & p).sum(0), FN=(t & ~p).sum(0)),
                        index=LABELS + ["None"])


def paired(gold14, base14, edit14) -> dict:
    gb, ge = grade(gold14, base14), grade(gold14, edit14)
    d = (ge.s - gb.s).to_numpy(float)
    mb, me = metrics(gold14, base14), metrics(gold14, edit14)
    return dict(n=len(d), mean_d=d.mean(), se=d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan"),
                d_micro=me["micro"] - mb["micro"], d_macro=me["macro"] - mb["macro"],
                micro_base=mb["micro"], micro_edit=me["micro"], macro_base=mb["macro"], macro_edit=me["macro"],
                **{f"{f}_base": int(gb[f].sum()) for f in ("F1", "F2", "F3")},
                **{f"{f}_edit": int(ge[f].sum()) for f in ("F1", "F2", "F3")},
                wrong_to_right=int((~gb.right & ge.right).sum()), right_to_wrong=int((gb.right & ~ge.right).sum()),
                changed_paragraphs=int((base14 != edit14).any(1).sum()))


def fmt_paired(r: dict) -> str:
    return (f"n={r['n']}  mean d={r['mean_d']:+.4f}  SE={r['se']:.4f}  (d/SE={r['mean_d'] / r['se'] if r['se'] else 0:+.2f})\n"
            f"micro {100 * r['micro_base']:.2f} -> {100 * r['micro_edit']:.2f} ({100 * r['d_micro']:+.2f})   "
            f"macro {100 * r['macro_base']:.2f} -> {100 * r['macro_edit']:.2f} ({100 * r['d_macro']:+.2f})\n"
            f"F1 {r['F1_base']}->{r['F1_edit']}  F2 {r['F2_base']}->{r['F2_edit']}  F3 {r['F3_base']}->{r['F3_edit']}   "
            f"wrong->right {r['wrong_to_right']}  right->wrong {r['right_to_wrong']}  changed {r['changed_paragraphs']}")


# ---------------------------------------------------------------- edits
def project_out(W, u, a):
    u = u.to(W.dtype).to(W.device)
    return W - a * torch.outer(u, u) @ W


def unit(v):
    v = torch.as_tensor(v, dtype=torch.float32)
    return v / v.norm()


def validate(base: dict, P: dict) -> dict:
    changed = {}
    for k, v in P.items():
        b = base[k]
        if v.shape != b.shape:
            raise ValueError(f"{k}: shape changed {tuple(b.shape)} -> {tuple(v.shape)}")
        if torch.equal(v, b):
            continue
        bf, vf = b.flatten().float(), v.flatten().float()
        if bf.norm() > 0:
            c = (vf @ bf) / (bf @ bf)
            if (vf - c * bf).norm() <= 1e-6 * vf.norm() and abs(c - 1) > 1e-6:
                raise ValueError(f"{k}: whole-tensor rescaling (x{c:.4f}) is not allowed")
        changed[k] = v.detach().cpu()
    extra = set(P) - set(base)
    if extra:
        raise ValueError(f"unknown / frozen tensors: {sorted(extra)[:3]}")
    return changed


# ---------------------------------------------------------------- sessions
class Session:
    def __init__(self, sid: str):
        self.dir = SR / "sessions" / sid
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "edits").mkdir(exist_ok=True)
        self.state_f = self.dir / "state.json"
        self.state = json.loads(self.state_f.read_text()) if self.state_f.exists() else \
            dict(code=0, gen_read=0, submitted=None, reserved=None, decision=None, read_used=[])

    def save(self):
        self.state_f.write_text(json.dumps(self.state, indent=1))

    def log(self, rec):
        with open(self.dir / "log.jsonl", "a") as f:
            f.write(json.dumps(rec | {"time": time.strftime("%H:%M:%S")}, default=str) + "\n")

    def edit_path(self, name):
        return self.dir / "edits" / f"{name}.pt"


def check_why(why: str):
    n = len([s for s in re.split(r"(?<=[.!?])\s+", why.strip()) if s])
    if not why.strip() or n > 2:
        raise SystemExit(f"rationale must be 1-2 sentences (got {n})")


def resolve_ids(spec: str, prac: pd.DataFrame, sess: Session) -> list[str]:
    """SPEC: 'diag' | 'nondiag' | 'unread' (practice ids not used by any READ) | path to a file of ids | id,id,..."""
    allp = set(prac.pid)
    if spec == "diag":
        ids = diag_ids()
    elif spec == "nondiag":
        ids = sorted(allp - set(diag_ids()))
    elif spec == "unread":
        ids = sorted(allp - set(sess.state["read_used"]))
    elif Path(spec).exists():
        ids = [l.strip() for l in Path(spec).read_text().split() if l.strip()]
    else:
        ids = spec.split(",")
    bad = [i for i in ids if i not in allp]
    if bad:
        raise PermissionError(f"{len(bad)} ids are not in PRACTICE (e.g. {bad[0]})")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sid")
    ap.add_argument("action", nargs="?")
    ap.add_argument("--why", default="")
    ap.add_argument("--file"); ap.add_argument("--ids"); ap.add_argument("--which", default="both")
    ap.add_argument("--edit"); ap.add_argument("--layers"); ap.add_argument("--decision")
    a = ap.parse_args()
    R = Runner()

    if a.sid == "diagnose":
        prac = pool_df("practice"); ids = diag_ids()
        df = prac.set_index("pid").loc[ids].reset_index()
        Y = np.stack(df.labels).astype(bool); P = R.predict(df)
        g, m = grade(Y, P), metrics(Y, P)
        out = pd.concat([df[["pid", "text"]], g], axis=1)
        out["gold"] = ["|".join(x) or "None" for x in df.label_names]
        out["pred"] = ["|".join(LABELS[i] for i in np.flatnonzero(p)) or "None" for p in P]
        out.drop(columns="text").to_csv(SR / "diagnosis.csv", index=False)
        print(f"DIAGNOSIS  base = {CKPT.name}  on {len(ids)} PRACTICE paragraphs")
        print(f"micro {100 * m['micro']:.2f}   macro {100 * m['macro']:.2f}   mean s {g.s.mean():+.3f}   exact-right {g.right.sum()}")
        for f, desc in (("F1", "pred frame, gold None"), ("F2", "pred None, gold frames"), ("F3", "|pred|>|gold|+1")):
            print(f"{f} ({desc}): {int(g[f].sum())}")
        print(per_label(Y, P).to_string())
        return

    S = Session(a.sid)
    prac = pool_df("practice").set_index("pid", drop=False)
    rec = dict(action=a.action, why=a.why)

    if a.action == "code":
        check_why(a.why)
        if S.state["code"] >= CODE_BUDGET:
            raise SystemExit("CODE budget exhausted")
        if S.state["submitted"]:
            raise SystemExit("session already submitted")
        name = Path(a.file).stem
        spec = importlib.util.spec_from_file_location(name, a.file); mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        base = R.editable(); P = {k: v.clone() for k, v in base.items()}
        H = argparse.Namespace(project_out=project_out, unit=unit,
                               load_read=lambda n: dict(np.load(S.dir / f"{n}.npz", allow_pickle=True)),
                               diagnosis=lambda: pd.read_csv(SR / "diagnosis.csv"))
        with torch.no_grad():
            mod.edit(P, H)
        S.state["code"] += 1; S.save()
        changed = validate(base, P)
        torch.save(changed, S.edit_path(name))
        info = {k: f"|Δ|/|W|={(changed[k].float() - base[k].cpu().float()).norm() / base[k].float().norm():.4f}" for k in changed}
        rec |= dict(file=a.file, edit=name, changed=info)
        print(f"edit '{name}' saved: {len(changed)} tensors changed   (CODE {S.state['code']}/{CODE_BUDGET})")
        for k, v in list(info.items())[:30]:
            print(f"  {k}  {v}")

    elif a.action in ("generate", "read"):
        check_why(a.why)
        if S.state["gen_read"] >= GEN_READ_BUDGET:
            raise SystemExit("GENERATE/READ budget exhausted")
        ids = resolve_ids(a.ids, prac, S); df = prac.loc[ids].reset_index(drop=True)
        changed = torch.load(S.edit_path(a.edit)) if a.edit else None
        S.state["gen_read"] += 1
        if a.action == "read":
            layers = [int(x) for x in a.layers.split(",")]
            R.load(changed); H = R.hidden(df, layers); R.load(None)
            n = S.state["gen_read"]
            np.savez(S.dir / f"read{n}.npz", ids=np.array(ids), **{f"L{l}": H[l] for l in layers})
            S.state["read_used"] = sorted(set(S.state["read_used"]) | set(ids))
            rec |= dict(ids=len(ids), layers=layers, file=f"read{n}")
            print(f"READ saved sr/sessions/{a.sid}/read{n}.npz  ({len(ids)} paragraphs x layers {layers})")
        else:
            Y = np.stack(df.labels).astype(bool)
            Pb = R.predict(df) if a.which in ("base", "both") else None
            if a.which in ("edit", "both"):
                R.load(changed); Pe = R.predict(df); R.load(None)
            if a.which == "both":
                r = paired(Y, Pb, Pe); rec |= dict(ids=len(ids), edit=a.edit, paired=r); print(fmt_paired(r))
            else:
                P = Pb if a.which == "base" else Pe; m = metrics(Y, P); g = grade(Y, P)
                rec |= dict(ids=len(ids), which=a.which, micro=m["micro"], macro=m["macro"])
                print(f"{a.which}: micro {100 * m['micro']:.2f} macro {100 * m['macro']:.2f}  "
                      f"F1 {g.F1.sum()} F2 {g.F2.sum()} F3 {g.F3.sum()}")
        S.save()
        print(f"(GENERATE/READ {S.state['gen_read']}/{GEN_READ_BUDGET})")

    elif a.action == "submit":
        if S.state["submitted"]:
            raise SystemExit("this session already used its RESERVED test")
        res = pool_df("reserved"); Y = np.stack(res.labels).astype(bool)
        Pb = R.predict(res); R.load(torch.load(S.edit_path(a.edit))); Pe = R.predict(res); R.load(None)
        r = paired(Y, Pb, Pe)
        S.state.update(submitted=a.edit, reserved=r); S.save()
        rec |= dict(edit=a.edit, reserved=r)
        print("RESERVED paired test (one-shot):\n" + fmt_paired(r))

    elif a.action == "decide":
        r = S.state["reserved"]
        if r is None:
            raise SystemExit("submit first")
        if a.decision == "accept" and not r["d_micro"] > 0:
            raise SystemExit("cannot accept: reserved micro-F1 did not go up")
        S.state["decision"] = a.decision; S.save()
        if a.decision == "accept":
            (SR / "model_in_use.json").write_text(json.dumps(dict(session=a.sid, edit=S.state["submitted"],
                                                                  path=str(S.edit_path(S.state['submitted'])))))
        rec |= dict(decision=a.decision)
        print(f"decision: {a.decision}")
    else:
        raise SystemExit(f"unknown action {a.action}")
    S.log(rec)


if __name__ == "__main__":
    main()
