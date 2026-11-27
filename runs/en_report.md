English only — official F1 ×100 (15 classes incl. None), mean ± std over 5 folds

| run | None | thr | EN micro | EN macro |
|---|---|---|---|---|
| phase1/A0_proto | implicit | global | 32.1 ± 2.0 | 22.3 ± 1.3 |
| phase1/A0_proto | implicit | EN | 30.6 ± 2.6 | 22.9 ± 2.0 |
| phase1/A0_proto | explicit | global | 29.4 ± 2.2 | 21.5 ± 1.0 |
| phase1/A0_proto | explicit | EN | 28.2 ± 3.2 | 22.7 ± 2.5 |
| phase1/A_linear | implicit | global | 33.3 ± 2.0 | 28.2 ± 1.1 |
| phase1/A_linear | implicit | EN | 33.3 ± 1.5 | 28.7 ± 1.5 |
| phase1/A_linear | explicit | global | 33.6 ± 3.9 | 27.9 ± 1.9 |
| phase1/A_linear | explicit | EN | 33.1 ± 3.1 | 27.8 ± 2.0 |
| phase1/A_linear_ctx | implicit | global | 32.3 ± 0.4 | 25.4 ± 2.6 |
| phase1/A_linear_ctx | implicit | EN | 33.4 ± 1.8 | 27.4 ± 4.0 |
| phase1/A_linear_ctx | explicit | global | 31.9 ± 1.3 | 25.5 ± 2.2 |
| phase1/A_linear_ctx | explicit | EN | 32.6 ± 1.9 | 25.9 ± 2.3 |
| phase1/B_ctx_lr3e-5 | implicit | global | 35.8 ± 2.4 | 30.0 ± 2.5 |
| phase1/B_ctx_lr3e-5 | implicit | EN | 34.1 ± 2.6 | 25.9 ± 5.5 |
| phase1/B_ctx_lr3e-5 | explicit | global | 34.9 ± 3.5 | 28.2 ± 3.3 |
| phase1/B_ctx_lr3e-5 | explicit | EN | 34.5 ± 4.0 | 27.7 ± 4.7 |
| phase1/B_ctx_lr5e-5 | implicit | global | 35.6 ± 3.8 | 28.6 ± 3.8 |
| phase1/B_ctx_lr5e-5 | implicit | EN | 33.9 ± 4.7 | 29.0 ± 4.3 |
| phase1/B_ctx_lr5e-5 | explicit | global | 34.6 ± 3.9 | 26.5 ± 3.4 |
| phase1/B_ctx_lr5e-5 | explicit | EN | 35.0 ± 4.6 | 27.8 ± 1.5 |
| phase1/B_tgt_lr3e-5 | implicit | global | 34.8 ± 3.5 | 28.5 ± 3.0 |
| phase1/B_tgt_lr3e-5 | implicit | EN | 35.8 ± 3.3 | 30.6 ± 3.8 |
| phase1/B_tgt_lr3e-5 | explicit | global | 34.4 ± 4.7 | 27.0 ± 4.1 |
| phase1/B_tgt_lr3e-5 | explicit | EN | 35.4 ± 4.8 | 29.1 ± 4.4 |
| phase1/B_tgt_lr5e-5 | implicit | global | 35.3 ± 3.6 | 27.3 ± 2.5 |
| phase1/B_tgt_lr5e-5 | implicit | EN | 35.1 ± 1.8 | 29.7 ± 3.5 |
| phase1/B_tgt_lr5e-5 | explicit | global | 35.8 ± 4.8 | 28.1 ± 3.4 |
| phase1/B_tgt_lr5e-5 | explicit | EN | 35.2 ± 3.8 | 28.0 ± 3.6 |
| phase1/C_xlmr_base | implicit | global | 35.5 ± 3.7 | 29.3 ± 5.6 |
| phase1/C_xlmr_base | implicit | EN | 35.3 ± 5.7 | 29.9 ± 6.7 |
| phase1/C_xlmr_base | explicit | global | 35.7 ± 5.5 | 28.6 ± 7.1 |
| phase1/C_xlmr_base | explicit | EN | 38.7 ± 3.1 | 31.4 ± 4.2 |
| phase1b/M_E1_extraEN | implicit | global | 36.8 ± 2.4 | 30.0 ± 3.3 |
| phase1b/M_E1_extraEN | implicit | EN | 35.8 ± 1.9 | 28.7 ± 2.4 |
| phase1b/M_E1_extraEN | explicit | global | 36.4 ± 3.0 | 29.2 ± 2.8 |
| phase1b/M_E1_extraEN | explicit | EN | 33.1 ± 1.7 | 26.9 ± 1.4 |
| phase1b/M_E3_extraEN_over2 | implicit | global | 36.2 ± 2.5 | 30.4 ± 1.5 |
| phase1b/M_E3_extraEN_over2 | implicit | EN | 34.3 ± 2.0 | 29.6 ± 0.9 |
| phase1b/M_E3_extraEN_over2 | explicit | global | 34.9 ± 3.5 | 28.6 ± 4.0 |
| phase1b/M_E3_extraEN_over2 | explicit | EN | 35.8 ± 3.6 | 28.0 ± 2.9 |
| phase1b/M_E3_extraEN_over2 [stage2] | implicit | EN | 33.8 ± 2.1 | 27.7 ± 1.5 |
| phase1b/M_E3_extraEN_over2 [stage2] | explicit | EN | 33.3 ± 3.3 | 27.1 ± 3.3 |

Per-label EN F1 for the best EN micro: phase1/C_xlmr_base (explicit, EN thr)

| label | P | R | F1 | gold | pred |
|---|---|---|---|---|---|
| Economic | 0.43 | 0.53 | 0.48 | 57 | 69 |
| Capacity_and_resources | 0.32 | 0.54 | 0.40 | 69 | 114 |
| Morality | 0.06 | 0.02 | 0.03 | 57 | 17 |
| Fairness_and_equality | 0.23 | 0.26 | 0.25 | 58 | 64 |
| Legality_Constitutionality_and_jurisprudence | 0.34 | 0.56 | 0.42 | 66 | 109 |
| Policy_prescription_and_evaluation | 0.30 | 0.49 | 0.37 | 74 | 121 |
| Crime_and_punishment | 0.35 | 0.47 | 0.40 | 75 | 100 |
| Security_and_defense | 0.33 | 0.53 | 0.41 | 164 | 262 |
| Health_and_safety | 0.29 | 0.55 | 0.38 | 76 | 145 |
| Quality_of_life | 0.37 | 0.52 | 0.43 | 90 | 128 |
| Cultural_identity | 0.16 | 0.09 | 0.12 | 32 | 19 |
| Public_opinion | 0.23 | 0.18 | 0.20 | 44 | 35 |
| Political | 0.36 | 0.58 | 0.44 | 159 | 260 |
| External_regulation_and_reputation | 0.33 | 0.41 | 0.36 | 69 | 86 |
| None | 0.44 | 0.40 | 0.42 | 229 | 207 |
