# Self-rewriting protocol on mCPT (English, Subtask 1) — report

## Setup
- Base: `ckpt/sr_base_M_E1.pt` = M_E1 recipe (mCPT + context + frame head, batch-1 EN) retrained WITHOUT the
  PRACTICE/RESERVED articles (the dev-37.7 checkpoint had memorised them). Thresholds frozen, None implicit.
- Pools (seed 0, by article, sha256 1e50e500…): PRACTICE 29 art / 314 para, RESERVED 13 / 122, HELDOUT = EN final 10 / 129.
- Edits: rank-1 projection `W - a u u^T W` on one layer's attention-output and FFN-output weights (u from READ
  contrasts of failing vs correct paragraphs), plus one constant shift of the pre-GELU FFN bias per session.
  Embeddings, final LayerNorm, head frozen; no whole-tensor rescaling. Budgets: CODE 200, GENERATE+READ 20 per session.
- Finding during Phase 0: shifting `output.dense.bias` / `attention.output.dense.bias` by a constant is a no-op
  (cancelled by the following LayerNorm), so the bias edit used `intermediate.dense.bias`.

## Phase 1 — 8 sessions (best try per session; all tries in sr/sessions/*/log.jsonl)
| session | target | hypothesis | best edit | practice Δmicro (d/SE) | reserved Δmicro | decision |
|---|---|---|---|---|---|---|
| s1 | F2 None-when-framed | a "missed frame" direction at the top layers suppresses frame scores | bias +0.05 | +0.65 (2.48) | −0.27 | withdraw |
| s2 | F1 frame-when-None | a "spurious frame" direction | L12 proj a=0.2 | +0.94 (1.99) | – | withdraw (<2 SE) |
| s3 | F3 too many labels | an "over-labelling" direction | L10 proj a=0.1 | +0.75 (1.83) | – | withdraw |
| s4 | FP Policy | Policy-trigger direction | L10 proj a=0.3 | +0.58 (1.78) | – | withdraw |
| s5 | FP Security | Security-trigger direction | L12 proj a=0.3 | +0.50 (1.83) | – | withdraw |
| s6 | FN Legality | Legality-suppression direction | bias +0.02 | +0.53 (1.997) | – | withdraw |
| s7 | FP Crime | Crime-trigger direction | L10 proj a=0.15 | +0.50 (1.61) | – | withdraw |
| s8 | FN Political | Political-suppression direction | bias +0.03 | +0.95 (3.04) | +0.21 (d/SE 0.68) | accept |

## Phase 2 — HELDOUT (run once)
| | micro | macro |
|---|---|---|
| base | 46.35 | 39.41 |
| base + s8 bias +0.03 | 44.04 | 37.67 |
| Δ (95% article-bootstrap CI) | −2.31 [−4.83, −0.68] | −1.74 [−3.48, +3.46] |
wrong→right 1, right→wrong 2, 17 of 129 paragraphs changed.

## Verdict (honest)
- No edit produced a reliable gain. Every projection gave a similar small positive practice bump (+0.2 to +0.9)
  regardless of its target, and the targeted error counts barely moved: a generic score shift, not a targeted fix.
- The only accepted edit (a global pre-GELU bias shift) passed RESERVED by +0.21 micro with d/SE 0.68, and then
  HURT on HELDOUT (−2.3 micro, CI excludes 0). The acceptance rule (reserved micro up, any amount) is too weak
  for 122 paragraphs; a margin or a significance requirement on RESERVED would have rejected it.
- With ~40 practice tests, a few passing 2·SE by chance is expected; s1 and s8 behave like that.
- Conclusion: gradient-free rank-1 / bias self-editing does not improve this mCPT frame classifier. The model in
  use should revert to the base. No dev upload of the edited model is recommended.
