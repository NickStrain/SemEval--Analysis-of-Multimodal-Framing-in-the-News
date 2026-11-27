# Step 3 — error analysis (mcpt_en, EN CV out-of-fold, official 15-class scoring)

Overall micro 36.82. *Micro cost* = micro-F1 gain if this label's FP and FN were all fixed.

| label | TP | FP | FN | precision | recall | micro cost |
|---|---|---|---|---|---|---|
| None | 82 | 129 | 147 | 0.39 | 0.36 | +10.62 |
| Security_and_defense | 71 | 130 | 93 | 0.35 | 0.43 | +7.53 |
| Political | 79 | 114 | 80 | 0.41 | 0.50 | +6.50 |
| Quality_of_life | 27 | 35 | 63 | 0.44 | 0.30 | +4.26 |
| Capacity_and_resources | 14 | 35 | 55 | 0.29 | 0.20 | +3.79 |
| Morality | 0 | 7 | 57 | 0.00 | 0.00 | +3.49 |
| Health_and_safety | 31 | 51 | 45 | 0.38 | 0.41 | +3.44 |
| External_regulation_and_reputation | 24 | 49 | 45 | 0.33 | 0.35 | +3.41 |
| Crime_and_punishment | 35 | 67 | 40 | 0.34 | 0.47 | +3.38 |
| Policy_prescription_and_evaluation | 46 | 105 | 28 | 0.30 | 0.62 | +3.23 |
| Legality_Constitutionality_and_jurisprudence | 26 | 56 | 40 | 0.32 | 0.39 | +3.21 |
| Fairness_and_equality | 8 | 14 | 50 | 0.36 | 0.14 | +3.18 |
| Public_opinion | 15 | 27 | 29 | 0.36 | 0.34 | +2.13 |
| Economic | 34 | 46 | 23 | 0.42 | 0.60 | +2.04 |
| Cultural_identity | 3 | 10 | 29 | 0.23 | 0.09 | +1.88 |

## None errors
- predicted a frame when gold is None: **147** of 229 None paragraphs
- predicted None when gold has frames: **129** of 678 framed paragraphs

## Micro-F1 by paragraph length (tokens)

| length | paragraphs | gold None % | micro |
|---|---|---|---|
| < 40 | 191 | 41% | 39.4 |
| 40-100 | 499 | 23% | 34.8 |
| > 100 | 217 | 16% | 39.1 |

## Top 4 labels by micro cost

1. None (+10.62)
2. Security_and_defense (+7.53)
3. Political (+6.50)
4. Quality_of_life (+4.26)
