# Final analyses (EN, 5-fold CV, nested; 86 articles / 907 paragraphs)

## A. Ablation on CV

| system | EN micro | per fold | EN macro |
|---|---|---|---|
| mCPT alone | 35.69 | [36.5, 33.4, 37.3, 35.9, 35.2] | 33.4 |
| XLM-R large alone | 39.01 | [40.4, 38.0, 39.4, 38.9, 38.2] | 37.7 |
| silver DeBERTa alone | 37.06 | [36.6, 32.9, 40.0, 40.7, 34.8] | 35.4 |
| silver DeBERTa, per-label thresholds | 34.57 | [33.2, 30.7, 36.8, 39.0, 32.7] | 31.4 |
| mCPT + XLM-R large | 38.82 | [39.8, 37.4, 37.9, 39.7, 39.0] | 37.1 |
| mCPT + XLM-R large + silver DeBERTa | 41.67 | [42.4, 39.6, 42.4, 42.0, 41.6] | 40.1 |
| mCPT + XLM-R large + silver DeBERTa + title rule | 42.74 | [44.6, 40.4, 43.0, 42.0, 43.2] | 40.7 |

## B. Paired bootstrap over articles (2000 resamples): gain of B over A, 95% CI, share of resamples with a gain

| A | B | gain | 95% CI | P(gain > 0) |
|---|---|---|---|---|
| silver DeBERTa, per-label thresholds | silver DeBERTa alone | +2.49 | [+0.73, +4.39] | 1.00 |
| mCPT alone | mCPT + XLM-R large | +3.13 | [+0.96, +5.31] | 1.00 |
| mCPT + XLM-R large | ensemble | +2.85 | [+1.78, +3.90] | 1.00 |
| mCPT alone | ensemble | +5.98 | [+3.94, +8.05] | 1.00 |
| mCPT + XLM-R large + silver DeBERTa | ensemble + title rule | +1.07 | [+0.22, +1.97] | 0.99 |

## C. Per-frame F1 of the final CV system, without / with the title rule

| frame | gold n | F1 | F1 + title rule |
|---|---|---|---|
| Economic | 57 | 0.46 | 0.47 |
| Capacity_and_resources | 69 | 0.40 | 0.40 |
| Morality | 57 | 0.09 | 0.09 |
| Fairness_and_equality | 58 | 0.35 | 0.35 |
| Legality_Constitutionality_and_jurisprudence | 66 | 0.51 | 0.50 |
| Policy_prescription_and_evaluation | 74 | 0.43 | 0.43 |
| Crime_and_punishment | 75 | 0.45 | 0.45 |
| Security_and_defense | 164 | 0.46 | 0.47 |
| Health_and_safety | 76 | 0.42 | 0.43 |
| Quality_of_life | 90 | 0.46 | 0.46 |
| Cultural_identity | 32 | 0.33 | 0.33 |
| Public_opinion | 44 | 0.44 | 0.43 |
| Political | 159 | 0.46 | 0.45 |
| External_regulation_and_reputation | 69 | 0.44 | 0.43 |
| None | 229 | 0.31 | 0.41 |

- without title rule: precision 0.388, recall 0.450, titles predicted None 40% (gold 56%), title micro 36.2, body micro 42.1
- with title rule: precision 0.400, recall 0.459, titles predicted None 93% (gold 56%), title micro 52.2, body micro 42.1

## D. EN_best_model on the 224 held-out 2b EN paragraphs (never trained on; used by the notebook for selection)

- micro 46.76 -> 48.38 with the title rule (3x threshold); titles 20, gold None among titles 80%

| frame | gold n | F1 | F1 + title rule |
|---|---|---|---|
| Economic | 8 | 0.57 | 0.57 |
| Capacity_and_resources | 20 | 0.14 | 0.14 |
| Morality | 5 | 0.22 | 0.22 |
| Fairness_and_equality | 20 | 0.00 | 0.00 |
| Legality_Constitutionality_and_jurisprudence | 19 | 0.55 | 0.55 |
| Policy_prescription_and_evaluation | 14 | 0.42 | 0.43 |
| Crime_and_punishment | 26 | 0.46 | 0.47 |
| Security_and_defense | 42 | 0.47 | 0.46 |
| Health_and_safety | 17 | 0.65 | 0.67 |
| Quality_of_life | 20 | 0.33 | 0.33 |
| Cultural_identity | 3 | 0.00 | 0.00 |
| Public_opinion | 9 | 0.56 | 0.56 |
| Political | 34 | 0.54 | 0.57 |
| External_regulation_and_reputation | 23 | 0.47 | 0.47 |
| None | 66 | 0.57 | 0.64 |
