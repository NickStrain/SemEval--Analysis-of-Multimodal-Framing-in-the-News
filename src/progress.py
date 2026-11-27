"""Progress bars for the Phase 1 queue, read from runs/phase1/<run>.log (does not touch the jobs).

    python -m src.progress [--phase phase1b]   # one snapshot (default phase1b)
    python -m src.progress --watch    # refresh every 10 s (Ctrl-C to stop)
"""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PHASE = sys.argv[sys.argv.index("--phase") + 1] if "--phase" in sys.argv else "phase1b"
P1 = ROOT / "runs" / PHASE
QUEUE = P1 / "queue.sh"
W = 30


def planned():
    """[(name, epochs)] from queue.sh."""
    out = []
    for line in QUEUE.read_text().splitlines():
        m = re.match(r"run (\S+) +(.*)", line)
        if not m:
            continue
        args = m.group(2)
        for var, val in re.findall(r'^(\w+)="(.*)"$', QUEUE.read_text(), re.M):
            args = args.replace("$" + var, val)
        ep = re.search(r"--epochs (\d+)", args)
        s2 = re.search(r"--stage2-en (\d+)", args)
        if ep:
            out.append((m.group(1), int(ep.group(1)) + (int(s2.group(1)) if s2 else 0)))
    return out


def fmt_t(s):
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def bar(frac):
    n = int(round(frac * W))
    return "█" * n + "░" * (W - n)


def snapshot():
    rows, total_left = [], 0.0
    sec_per_epoch_seen = {}
    for name, epochs in planned():
        log = P1 / f"{name}.log"
        total = 5 * epochs
        recs = []
        if log.exists():
            recs = [json.loads(l) for l in log.read_text().splitlines() if l.startswith("{")]
        done = len(recs)
        finished = (P1 / name / "oof.npz").exists() and done >= total
        failed = log.exists() and "Error" in log.read_text()[-3000:] and not finished
        spent = sum(r["sec"] for r in recs)
        spe = spent / done if done else None
        if spe:
            sec_per_epoch_seen[name] = spe
        if finished:
            status, eta = "done", fmt_t(spent)
        elif failed:
            status, eta = "FAILED", "-"
        elif done or log.exists():
            left = (total - done) * (spe or 0)
            total_left += left
            last = recs[-1] if recs else {}
            status = f"fold {last.get('fold', 0) + 1}/5 ep {last.get('epoch', 0)}/{epochs}" if recs else "loading"
            eta = f"ETA {fmt_t(left)}" if spe else "ETA ?"
        else:
            status, eta = "queued", ""
        rows.append((name, done / total, f"{done}/{total} ep", status, eta, finished))
    # queued runs: estimate from similar finished/running runs
    est = {"mcpt": 18, "ctx": 45, "base": 50, "large": 100, "M_": 45, "X_": 50}
    for name, epochs in planned():
        log = P1 / f"{name}.log"
        if not log.exists():
            k = next((p for p in ("M_", "X_") if name.startswith(p)), None) or \
                ("large" if "large" in name else "base" if "base" in name else "ctx" if "ctx" in name else "mcpt")
            total_left += 5 * epochs * est[k]
    lines = [f"{PHASE} training queue   {time.strftime('%H:%M:%S')}", ""]
    for name, frac, cnt, status, eta, _ in rows:
        lines.append(f"{name:<19} {bar(frac)} {100 * frac:5.1f}%  {cnt:>9}  {status:<18} {eta}")
    fin = sum(r[5] for r in rows)
    lines += ["", f"runs done {fin}/{len(rows)}   estimated time left for whole queue: ~{fmt_t(total_left)}"]
    return "\n".join(lines)


if __name__ == "__main__":
    if "--watch" in sys.argv:
        try:
            while True:
                print("\033[2J\033[H" + snapshot(), flush=True)
                time.sleep(10)
        except KeyboardInterrupt:
            pass
    else:
        print(snapshot())
