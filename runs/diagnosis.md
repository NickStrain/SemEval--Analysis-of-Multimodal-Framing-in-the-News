# Diagnosis of the best EN system (mCPT + XLM-R large + silver DeBERTa)

Nested micro 41.62 ± 1.04; thresholds chosen per fold: [0.25, 0.25, 0.25, 0.25, 0.25]

## 1. Data
- EN training paragraphs: 907 in 86 articles (tiny for a 15-way multi-label task)
- gold labels per paragraph: mean 1.20; distribution 0:25% 1:41% 2:25% 3+:9%
- predicted labels per paragraph: mean 1.56; predicted None 12% vs gold None 25%

## 2. Is the model ranking frames badly, or deciding badly?
| label | gold n | AUC | AP | F1 (nested) |
|---|---|---|---|---|
| Economic | 57 | 0.92 | 0.48 | 0.46 |
| Capacity_and_resources | 69 | 0.87 | 0.39 | 0.40 |
| Morality | 57 | 0.74 | 0.14 | 0.09 |
| Fairness_and_equality | 58 | 0.86 | 0.32 | 0.35 |
| Legality_Constitutionality_and_jurisprudence | 66 | 0.90 | 0.46 | 0.51 |
| Policy_prescription_and_evaluation | 74 | 0.87 | 0.40 | 0.43 |
| Crime_and_punishment | 75 | 0.88 | 0.42 | 0.45 |
| Security_and_defense | 164 | 0.77 | 0.40 | 0.46 |
| Health_and_safety | 76 | 0.85 | 0.37 | 0.42 |
| Quality_of_life | 90 | 0.86 | 0.45 | 0.46 |
| Cultural_identity | 32 | 0.91 | 0.24 | 0.33 |
| Public_opinion | 44 | 0.88 | 0.33 | 0.44 |
| Political | 159 | 0.80 | 0.45 | 0.46 |
| External_regulation_and_reputation | 69 | 0.89 | 0.43 | 0.44 |

## 3. Upper bounds: what each sub-problem is worth if solved (cheating on purpose)
| scenario | micro |
|---|---|
| **current system (honest)** | **41.6** |
| threshold tuned on the test fold itself | 42.3 |
| per-label thresholds tuned on test fold | 45.1 |
| None decision perfect (frames: best score if empty) | 57.0 |
| number of labels known, top-k by score | 59.5 |
| perfect None + top-1 frame only | 51.0 |
| article's gold frame set known (+ test-fold thr) | 47.7 |
| predict ALL article gold frames on every paragraph | 33.9 |

## 4. Error budget (nested predictions)
Totals: TP 593, FP 934, FN 726 -> precision 0.39, recall 0.45
| label | TP | FP | FN | micro gain if fixed |
|---|---|---|---|---|
| Security_and_defense | 101 | 177 | 63 | +6.4 |
| None | 53 | 59 | 176 | +10.2 |
| Political | 77 | 97 | 82 | +6.0 |
| Quality_of_life | 50 | 76 | 40 | +3.4 |
| Policy_prescription_and_evaluation | 41 | 77 | 33 | +3.0 |
| Health_and_safety | 40 | 73 | 36 | +3.1 |
| External_regulation_and_reputation | 40 | 72 | 29 | +2.7 |
| Crime_and_punishment | 40 | 61 | 35 | +2.9 |
| Capacity_and_resources | 31 | 56 | 38 | +3.0 |
| Morality | 4 | 30 | 53 | +3.4 |
| Legality_Constitutionality_and_jurisprudence | 41 | 54 | 25 | +2.2 |
| Economic | 30 | 43 | 27 | +2.1 |
| Fairness_and_equality | 17 | 22 | 41 | +2.6 |
| Public_opinion | 19 | 23 | 25 | +1.7 |
| Cultural_identity | 9 | 14 | 23 | +1.5 |

## 5. Most common confusions (predicted X where gold has Y but not X)
- predicted Security_and_defense instead of Political: 23
- predicted Policy_prescription_and_evaluation instead of Political: 17
- predicted External_regulation_and_reputation instead of Political: 15
- predicted Security_and_defense instead of Health_and_safety: 12
- predicted Crime_and_punishment instead of Security_and_defense: 11
- predicted Security_and_defense instead of Quality_of_life: 10
- predicted Political instead of Fairness_and_equality: 9
- predicted Security_and_defense instead of External_regulation_and_reputation: 8
- predicted Legality_Constitutionality_and_jurisprudence instead of Morality: 8
- predicted Security_and_defense instead of Fairness_and_equality: 8

## 6. By paragraph type
- title (para 0): 86 paras, gold None 56%, pred None 40%, micro 36.2
- body: 821 paras, gold None 22%, pred None 10%, micro 42.1
- <25 words: 172 paras, gold None 45%, pred None 35%, micro 37.6
- 25-80 words: 586 paras, gold None 22%, pred None 8%, micro 41.1
- >=80 words: 149 paras, gold None 17%, pred None 3%, micro 46.3

## 7. Context
- adjacent paragraphs: mean label Jaccard 0.19; a paragraph's frames appear elsewhere in its article 100% of the time (by definition 100%); article has 5.0 frames on average
