"""Check a prediction file with the official scorer (format check) and zip it for Codabench (file at zip root)."""
import subprocess
import sys
import zipfile
from pathlib import Path

from src.data import LABELS, ROOT

pred = Path(sys.argv[1]); lang = sys.argv[2] if len(sys.argv) > 2 else "en"
part = ROOT / "data" / "3_dev-set-nov-release" / lang.upper() / "subtask-1-participant.txt"
keys = lambda f: [tuple(x.strip() for x in l.rstrip("\r\n").split("\t")[:3]) for l in open(f, encoding="utf-8-sig") if l.strip()]
assert keys(pred) == keys(part), "rows/keys differ from the participant file"
frames = ROOT / "runs" / "submissions" / "frames.txt"; frames.write_text("\n".join(LABELS) + "\n")
scorer = ROOT / "starter-kit/starter-kit-v2/subtask1/scorer-subtask-1.py"
r = subprocess.run([sys.executable, str(scorer), "-p", str(pred), "-f", str(frames)], capture_output=True, text=True)
print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr)
assert "format is correct" in r.stdout, "official format check failed"
z = pred.with_suffix(".zip")
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(pred, arcname=pred.name)
print(f"zip ready: {z}  ({len(keys(pred))} rows, contains {zipfile.ZipFile(z).namelist()})")
