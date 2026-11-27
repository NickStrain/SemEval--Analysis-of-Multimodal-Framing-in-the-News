"""Load the user's EN_best_model bundle (paraphrase-multilingual-mpnet-base-v2 encoder + MLP head) and score paragraphs.
Checks that it reproduces the bundle's own dev predictions. Saves dev probabilities (558,14) in OUR label order."""
import numpy as np, torch, torch.nn as nn, json
from transformers import AutoModel, AutoTokenizer
from src.data import LABELS, ROOT
from src.final_model import load_dev
B = ROOT / "ckpt/EN_best_model"; meta = json.load(open(B / "meta.json"))
tok = AutoTokenizer.from_pretrained(B / "encoder"); enc = AutoModel.from_pretrained(B / "encoder").cuda().eval()
sd = torch.load(B / "head.pt", map_location="cuda", weights_only=True)
dev = load_dev("en"); ref = [set(x for x in l.rstrip("\n").split("\t")[3:] if x) for l in open(B / "3_dev-set-nov-release__EN__subtask-1-predictions.txt")]
@torch.no_grad()
def emb(texts):
    out = []
    for b in range(0, len(texts), 32):
        t = tok(texts[b:b + 32], padding=True, truncation=True, max_length=meta["max_len"], return_tensors="pt").to("cuda")
        h = enc(**t).last_hidden_state; m = t["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).float())
    return torch.cat(out)
Z = emb(dev.text.tolist())
best = None
for act in (nn.ReLU(), nn.GELU(), nn.Tanh()):
    head = nn.Sequential(nn.Linear(768, 256), act, nn.Dropout(0.5), nn.Linear(256, 14)).cuda().eval(); head.load_state_dict(sd)
    for norm in (False, True):
        with torch.no_grad():
            p = torch.sigmoid(head(nn.functional.normalize(Z, dim=-1) if norm else Z)).cpu().numpy()
        pred = [{meta["labels"][j] for j in np.flatnonzero(r >= meta["threshold"])} for r in p]
        agree = np.mean([a == b for a, b in zip(pred, ref)])
        print(f"{type(act).__name__:5s} l2norm={norm}: exact match with bundle predictions {100 * agree:.1f}%")
        if best is None or agree > best[0]:
            best = (agree, p)
P = best[1][:, [meta["labels"].index(l) for l in LABELS]]
np.save(ROOT / "runs/submissions/EN_best_model_EN_dev_scores.npy", P); print("saved dev probs, match", round(100 * best[0], 1))
