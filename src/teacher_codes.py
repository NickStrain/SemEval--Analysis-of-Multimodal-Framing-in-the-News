"""Convert compact teacher lines 'pid|CODE:p CODE:p|none|why' into JSONL (teacher/labels/<name>.jsonl)."""
import json
import sys
from pathlib import Path

CODES = {"ECO": "Economic", "CAP": "Capacity_and_resources", "MOR": "Morality", "FAI": "Fairness_and_equality",
         "LEG": "Legality_Constitutionality_and_jurisprudence", "POL": "Policy_prescription_and_evaluation",
         "CRI": "Crime_and_punishment", "SEC": "Security_and_defense", "HEA": "Health_and_safety",
         "QOL": "Quality_of_life", "CUL": "Cultural_identity", "PUB": "Public_opinion", "POLIT": "Political",
         "EXT": "External_regulation_and_reputation"}

src = Path(sys.argv[1]); out = []
for line in src.read_text().splitlines():
    if not line.strip():
        continue
    pid, frames, none, why = [x.strip() for x in line.split("|")]
    p = {CODES[c]: float(v) for c, v in (t.split(":") for t in frames.split())} if frames else {}
    assert len(why.split()) <= 15, (pid, why)
    out.append(json.dumps({"pid": pid, "p": p, "none": float(none), "why": why}))
dst = src.with_suffix(".jsonl"); dst.write_text("\n".join(out) + "\n")
print(f"{dst}: {len(out)} paragraphs")
