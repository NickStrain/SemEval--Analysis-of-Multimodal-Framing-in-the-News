"""Phase 0.2: load mCPT, dump config / params / vocab / architecture to runs/model_info.json.

The HF repo ships only config.json + pytorch_model.bin (no tokenizer). Its config
(BERT, 12 layers, hidden 384, vocab 250037) matches
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, so we use that tokenizer and
assert the vocab sizes agree.
"""
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

MODEL = "socialcomplab/mcpt-body-semeval2023task3"
TOKENIZER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUT = Path(__file__).resolve().parent.parent / "runs" / "model_info.json"

model = AutoModel.from_pretrained(MODEL)
tok = AutoTokenizer.from_pretrained(TOKENIZER)
cfg = model.config
# Multilingual-MiniLM ships a 250037-row embedding matrix with the 250002-token XLM-R
# tokenizer: rows 250002..250036 are padding that no token id ever reaches.
assert len(tok) <= cfg.vocab_size, (len(tok), cfg.vocab_size)

n_params = sum(p.numel() for p in model.parameters())
n_emb = model.embeddings.word_embeddings.weight.numel()
info = {
    "model": MODEL,
    "tokenizer": TOKENIZER,
    "base_architecture": f"{cfg.model_type} / {type(model).__name__}",
    # transformers>=5 overwrites _name_or_path on load, so read it from the raw config file.
    "parent_checkpoint": json.load(open(hf_hub_download(MODEL, "config.json")))["_name_or_path"],
    "lineage": "microsoft/Multilingual-MiniLM-L12-H384 -> paraphrase-multilingual-MiniLM-L12-v2 (assumed, from matching config) -> Kwin-exe/multilang-body-wcsl-randomsampling -> mCPT",
    "num_parameters": n_params,
    "num_parameters_word_embeddings": n_emb,
    "num_parameters_non_embedding": n_params - n_emb,
    "vocab_size_config": cfg.vocab_size,
    "vocab_size_tokenizer": len(tok),
    "unused_embedding_rows": cfg.vocab_size - len(tok),
    "head": "none (feature-extraction encoder; classification head must be trained)",
    "config": cfg.to_dict(),
}
print(cfg)
print(f"architecture: {info['base_architecture']}  (parent: {info['parent_checkpoint']})")
print(f"params: {n_params:,}  (word embeddings {n_emb:,}, rest {n_params - n_emb:,})")
print(f"vocab: config {cfg.vocab_size}, tokenizer {len(tok)}")
OUT.write_text(json.dumps(info, indent=1, default=str))
print(f"wrote {OUT}")
