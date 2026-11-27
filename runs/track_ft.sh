#!/usr/bin/env bash
# Fine-tuning the COPY of EN_best_model.   Usage: watch -n 20 bash runs/track_ft.sh [run]
cd "$(dirname "$0")/.."; R=${1:-ft1}; L=runs/ext_ft_$R.log
[ -f $L ] || { echo "no log yet: $L"; exit; }
python3 - "$L" "$R" <<'PY'
import json, sys, time
L, R = sys.argv[1], sys.argv[2]
ev = [json.loads(l) for l in open(L) if l.strip()]
EPOCHS = next((e.get("epochs") for e in ev if "epochs" in e), None) or 8
bar = lambda f, w=30: "[" + "#" * int(round(f * w)) + "-" * (w - int(round(f * w))) + f"] {100 * f:5.1f}%"
evals = [e for e in ev if e["event"] == "eval"]; saves = [e for e in ev if e["event"] == "saved_best"]
last = ev[-1]; done = last["event"] == "done"
print(time.strftime("%H:%M:%S"), f"| run {R}", "| FINISHED" if done else "| running")
print("\nEN score (micro-F1, official 15 classes) on the 224 held-out English 2b paragraphs")
ref = evals[0]["val_micro"] if evals else None
for e in evals:
    tag = "your original model" if e["epoch"] == 0 else f"{e['val_micro'] - ref:+.2f} vs original"
    print(f"  epoch {e['epoch']}: EN {e['val_micro']:5.2f}  (thr {e['thr']:.2f})  {tag}" + ("   <- NEW BEST, saved" if e.get("improved") else ""))
if evals:
    b = max(evals, key=lambda e: e["val_micro"]); print(f"  BEST so far: epoch {b['epoch']}  EN {b['val_micro']:.2f}")
ep_done = max([e["epoch"] for e in evals] + [0])
cur = [e for e in ev if e["event"] == "progress" and e["epoch"] == ep_done + 1]
frac_ep = cur[-1]["batch"] / cur[-1]["of"] if cur and not done else (1.0 if done else 0.0)
print("\nprogress")
if not done:
    print(f"  epoch {ep_done + 1}/{EPOCHS} {bar(frac_ep)}" + (f"  loss {cur[-1]['loss']}" if cur else ""))
print(f"  overall     {bar(min(1.0, (ep_done + (0 if done else frac_ep)) / EPOCHS))}")
print("\nsaved best models (each in its own folder, original untouched)")
for s in saves: print(f"  {s['folder']}   zip: {s['zip']}   format ok: {s['format_ok']}")
if not saves: print("  none yet (saved only when the EN score beats every earlier epoch)")
PY
echo; (cd ckpt/EN_best_model && sha256sum -c --quiet ../EN_best_model.sha256 && echo "original ckpt/EN_best_model untouched: yes (checksums match)")
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed 's/^/GPU: /'
