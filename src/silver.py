"""Step 6.1-6.2: silver data from Arora et al. 2025 (copenlu/mm-framing, framing_subset.csv, MIT).

The corpus has NO article body text. We use 'text-topic-exp' (a 2-3 sentence GPT summary of the article content)
as a paragraph-like input, with the article-level 'text-generic-frame' labels. 'text-generic-frame-exp' is NOT
used (it explains the frames, i.e. leaks the labels). Output: data/silver/silver_en.parquet (title, text, y[14]).
"""
import ast
import json

import numpy as np
import pandas as pd

from src.data import LABEL2ID, ROOT

MAP = {"economic": "Economic", "cap&res": "Capacity_and_resources", "morality": "Morality",
       "fairness": "Fairness_and_equality", "legality": "Legality_Constitutionality_and_jurisprudence",
       "policy": "Policy_prescription_and_evaluation", "crime": "Crime_and_punishment",
       "security": "Security_and_defense", "health": "Health_and_safety", "quality_life": "Quality_of_life",
       "culture": "Cultural_identity", "public_op": "Public_opinion", "political": "Political",
       "regulation": "External_regulation_and_reputation"}

if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data/silver/framing_subset.csv", usecols=["uuid", "title", "text-topic-exp", "text-generic-frame"])
    df = df.dropna(subset=["text-topic-exp", "text-generic-frame"])
    df["text"] = df["text-topic-exp"].str.replace(r"\s*This is a clear indication that the topic.*$", "", regex=True).str.strip()
    Y = np.zeros((len(df), 14), np.int8); unk = set()
    for i, s in enumerate(df["text-generic-frame"]):
        for f in ast.literal_eval(s):
            if f in MAP:
                Y[i, LABEL2ID[MAP[f]]] = 1
            else:
                unk.add(f)
    keep = (Y.sum(1) > 0) & (df.text.str.len() > 40).values
    out = pd.DataFrame({"title": df.title.fillna("").values[keep], "text": df.text.values[keep]})
    out["y"] = list(Y[keep])
    out = out.sample(n=min(150_000, len(out)), random_state=0).reset_index(drop=True)
    out.to_parquet(ROOT / "data/silver/silver_en.parquet")
    (ROOT / "configs/silver_mapping.json").write_text(json.dumps(MAP, indent=1))
    print(f"silver rows {len(out)} (unknown frame names: {unk or 'none'}); labels/row {np.stack(out.y).sum(1).mean():.2f}")
    print("example:", out.title[0][:80], "||", out.text[0][:200], "||", [k for k, v in LABEL2ID.items() if out.y[0][v]])
