"""Machine-translate non-English GOLD paragraphs to English (facebook/nllb-200-distilled-600M, free, HF).

Sources: CV rows in BG/PT/RU (src_fold = their CV fold), dev-release 2b gold BG/PT, batch-1 RU (src_fold = -1).
title / text / prev_text / next_text are translated sentence by sentence (no truncation of long paragraphs).
Output: data/trans/trans_en.parquet with the load_extra() columns + src_fold, lang='en', article_id 'TR_<orig>'.
Never touches the locked final split (load_cv excludes it).
"""
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.data import ROOT, _load_2b_gold, load_cv, load_extra

MODEL = "facebook/nllb-200-distilled-600M"
CODE = {"bg": "bul_Cyrl", "pt": "por_Latn", "ru": "rus_Cyrl"}

cv = load_cv(); cv = cv[cv.lang != "en"].copy(); cv["src_fold"] = cv.fold.astype(int); cv["mask_none"] = False
parts = [cv, _load_2b_gold("bg"), _load_2b_gold("pt"), load_extra(("ru",))]
df = pd.concat(parts, ignore_index=True)
df["src_fold"] = df["src_fold"].fillna(-1).astype(int)
print("rows to translate:", df.lang.value_counts().to_dict())

split = lambda t: [s for s in re.split(r"(?<=[.!?…»”\"])\s+", t) if s.strip()] if t else []
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
cache = {}
t0 = time.time()
for lang, code in CODE.items():
    sents = sorted({s for c in ("title", "text", "prev_text", "next_text") for t in df[df.lang == lang][c] for s in split(t)}
                   - cache.keys(), key=len)
    tok.src_lang = code
    print(f"{lang}: {len(sents)} unique sentences", flush=True)
    for b in range(0, len(sents), 48):
        batch = sents[b:b + 48]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=200).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"), max_new_tokens=256, num_beams=2)
        for s, o in zip(batch, tok.batch_decode(out, skip_special_tokens=True)):
            cache[s] = o
        if b % 960 == 0:
            print(f"  {b}/{len(sents)}  {(time.time() - t0) / 60:.1f} min", flush=True)
tr = lambda t: " ".join(cache[s] for s in split(t)) if t else ""
out = df[["article_id", "para_idx", "lang", "labels", "mask_none", "src_fold"]].copy() if "para_idx" in df else None
out = pd.DataFrame(dict(article_id="TR_" + df.article_id.astype(str), src_lang=df.lang, lang="en",
                        title=df.title.map(tr), text=df.text.map(tr), prev_text=df.prev_text.map(tr),
                        next_text=df.next_text.map(tr), labels=[np.asarray(y, np.int8).tolist() for y in df.labels],
                        mask_none=df.mask_none.fillna(False).astype(bool), src_fold=df.src_fold))
Path(ROOT / "data/trans").mkdir(exist_ok=True)
out.to_parquet(ROOT / "data/trans/trans_en.parquet")
print(f"saved data/trans/trans_en.parquet: {len(out)} rows in {(time.time() - t0) / 60:.1f} min")
for i in range(3):
    print("SRC:", df.text.iloc[i * 700][:150], "\nEN :", out.text.iloc[i * 700][:150])
