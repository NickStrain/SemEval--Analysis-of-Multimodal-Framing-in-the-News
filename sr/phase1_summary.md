# Phase 1 — 8 sessions (base: ckpt/sr_base_M_E1.pt; practice deltas on unread PRACTICE paragraphs)

| session | target | layer | a / bias | practice Δmicro | practice d/SE | reserved Δmicro | decision |
|---|---|---|---|---|---|---|---|
| s1 | F2 | all (bias) | +0.05 | +0.65 | 2.48 | −0.27 | withdraw (reserved fell) |
| s2 | F1 | 12 | 0.2 | +0.94 | 1.99 | – | withdraw (< 2 SE) |
| s3 | F3 | 10 | 0.1 | +0.75 | 1.83 | – | withdraw (< 2 SE) |
| s4 | FP Policy | 10 | 0.3 | +0.58 | 1.78 | – | withdraw (< 2 SE) |
| s5 | FP Security | 12 | 0.3 | +0.50 | 1.83 | – | withdraw (< 2 SE) |
| s6 | FN Legality | all (bias) | +0.02 | +0.53 | 1.997 | – | withdraw (< 2 SE, by 0.00007) |
| s7 | FP Crime | 10 | 0.15 | +0.50 | 1.61 | – | withdraw (< 2 SE) |
| s8 | FN Political | all (bias) | +0.03 | +0.95 | 3.04 | +0.21 (d/SE 0.68, macro −0.25) | **accept** |

Best try per session shown; all tries are in sr/sessions/<sid>/log.jsonl and sr/sessions_4_8.log.
Bias edits shift intermediate.dense.bias (pre-GELU) in all 12 layers; output-side bias shifts are no-ops under LayerNorm.
Model in use after Phase 1: base + s8_bias+0.03.
