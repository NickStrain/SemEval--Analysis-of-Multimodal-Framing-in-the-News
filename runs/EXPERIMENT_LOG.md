# Experiment log — SemEval-2027 Task 9, Subtask 1 (English focus)

Factual record of every experiment: setup, numbers, files, and how to reproduce. Raw sources:
`runs/log.csv` (one row per result), `runs/autopilot.md` (chronological notes), `runs/diagnosis.md`,
`runs/errors.md`, `runs/en_report.md`, `runs/phase1b/report.md`, `sr/report.md`, `runs/step10_0545_best41.6.json`.
Last updated 2026-11-26 ~20:30.

---

## 1. Task, data, evaluation

**Task.** Paragraph-level multi-label classification into 14 frames (Boydstun et al. policy frames); a paragraph
with no frame counts as the extra class "None". Languages BG, EN, PT, RU; this work targets **English**.

**Official metric.** Micro-F1 over 15 classes (14 frames + None), zero_division=1. `src/eval.py` and `src/score.py`
were checked to give identical results to the starter-kit scorer (`runs/verify_eval/`, 3 synthetic EN files).

**Data used (counts = paragraphs).**

| source | BG | EN | PT | RU | notes |
|---|---|---|---|---|---|
| Task-9 training (CV pool) | 780 | 907 | 1180 | 398 | 313 articles; EN = 86 articles |
| Locked final split | | | | | 35 articles, never used for selection (`data/splits.json`) |
| Batch-1 extra (SemEval-2023 style files) | – | 252 | – | 676 | only framed paragraphs listed → None output masked |
| Dev-release Subtask-2b gold (ST1 labels) | 180 | 224 | 240 | – | different articles from the ST1 dev set |
| Silver: copenlu/mm-framing (Arora et al. 2025) | – | 150,000 | – | – | GPT summaries + article-level frames (`src/silver.py`) |
| NLLB-200-600M translations to EN | ← 3,454 rows from BG/PT/RU gold → | | | | `src/translate.py`, `data/trans/trans_en.parquet` |
| ST1 dev set (unlabelled, Codabench) | | 558 | | | 55 documents |

**Label statistics (EN CV pool).** 1.20 labels/paragraph; 25% None, 41% one frame, 25% two, 9% three or more.
Titles (paragraph 0): 56% None. Rarest frames: Cultural_identity 32, Public_opinion 44, Morality 57.

**Protocol.**
- 5-fold cross-validation by article, stratified by language (`data/folds.json`); all numbers below are EN rows only.
- "Nested": epoch, ensemble weights and thresholds for test fold k are chosen on the other 4 folds only.
- Keep rule (overnight plan): CV micro +≥1.0 and gain > std of paired per-fold differences.
- Dev (Codabench) scores were obtained only by uploading; dev labels were never available locally.

---

## 2. Cross-validation results (EN micro-F1 ×100, mean ± std over 5 folds)

### 2.1 Baselines and single models

| model | setup | EN micro | EN macro | file |
|---|---|---|---|---|
| mCPT prototypes (A0) | cosine to label prototypes, no training | 32.1 | 22.3 | runs/phase1/A0_proto |
| mCPT + logistic regression (A) | frozen embeddings | 33.3 | 28.2 | runs/phase1/A_linear |
| mCPT fine-tuned (M_E1) | ctx (title+prev+next), lr 5e-5, 12 ep, + batch-1 EN | 36.82 ± 1.46 | 32.0 | runs/phase1b/M_E1_extraEN |
| XLM-R base (C) | lr 2e-5 | 36.36 ± 4.29 | – | runs/phase1/C_xlmr_base |
| XLM-R large (XL_E1) | lr 1e-5, 8 ep, frozen emb, grad ckpt, + extra EN | 38.97 ± 0.89 | 35.5 | runs/phase1c/XL_E1 |
| DeBERTa-v3-large, gold only | BCE, 5 ep, EN only | 23.4–25.0 | 12–15 | runs/step4/deb_bce |
| DeBERTa-v3-large, silver → gold | 1 ep silver (150K) then 5 ep gold | 37.01 ± 2.99 | 33.9 | runs/step4/deb_silver_bce |

(mCPT/XLM-R numbers with a single global threshold; protocol details in `src/score_mcpt.py` / `src/step10.py`.)

### 2.2 Decoding / post-processing (Step 2, Step 9)

| variant | EN micro | kept? |
|---|---|---|
| per-label thresholds (0.05 grid), nested | e.g. DeBERTa-silver 34.47 | baseline |
| **one global threshold**, nested | DeBERTa-silver 37.01; holds for every model | **yes** |
| None rule (None if max prob < m) + cap k labels | −0.4 to −1.8 vs baseline | no |
| rank-dependent thresholds t1≥t2≥t3 | 41.33 vs 41.62 | no |
| LightGBM co-occurrence stacker (neighbour features) | 34.07–37.56 vs 37.5–41.6 | no |
| predicted label count (0/1/2/3+) + top-k | 36.37 (count accuracy 39%, below majority 41%) | no |
| None-gate classifier | 40.95 vs 41.62 | no |
| **title rule** (titles need 3× threshold; multiplier nested) | **42.63 ± 1.39 vs 41.62** (4/5 folds up) | **yes** |

### 2.3 Ensembles (weights + global threshold nested, EN)

| ensemble | EN micro | EN macro |
|---|---|---|
| mCPT + XLM-R base (per-label thr) | 37.52 | 31.1 |
| mCPT + XLM-R large | 38.93 | 35.0 |
| mCPT + XLM-R base + silver DeBERTa | 38.64 | 34.4 |
| **mCPT + XLM-R large + silver DeBERTa** | **41.62 ± 1.04** (folds 42.4 39.6 42.4 42.0 41.6) | 37.7 |
| same + title rule | 42.63 ± 1.39 | – |

### 2.4 Improving mCPT itself (all from the original mCPT weights; EN text unless stated)

| run | change | EN micro alone | in ensemble* |
|---|---|---|---|
| M_E1 | original recipe | 36.82 ± 1.46 | 39.22 |
| S1 | + 1 ep silver BCE stage (encoder) before gold | 35.65 ± 3.06 | 38.84 |
| S3 | train on EN text only: EN gold + extra EN + 3,454 translations | 36.31 ± 3.51 | 40.52 |
| C1 | BCE + 0.5 × label-aware contrastive loss (Jaccard-weighted, None as a label, τ 0.1, bs 32) | 37.48 ± 2.76 | 38.87 |
| C2 | 3 ep contrastive only, then 12 ep BCE (paper's two-stage order) | 35.39 ± 3.73 | 40.16 |
| C3 | 3 ep contrastive, then BCE + contrastive | 35.30 ± 2.30 | 40.31 |
| avg(S1,S3,C1,C3) | probability average of variants | 39.14 | – |
| avg(S1,C2,C3) | | 37.82 | 41.55 |

*ensemble = this mCPT + XLM-R large + silver DeBERTa, scored with `src/score_mcpt.py` (epoch chosen with a global
threshold, so the mCPT member differs from 2.3; compare within this table). Averages were picked from 63
combinations on the same folds → optimistic by roughly 0.5–1.

### 2.5 Other experiments

| experiment | result | where |
|---|---|---|
| Step 7 LLM teacher pilot (Claude, frame definitions, 260 EN paragraphs) | teacher micro 32.2 / macro 31.2; mCPT on same paragraphs 29.3; best ensemble 38.0 | src/teacher.py |
| Self-rewriting (gradient-free rank-1 / bias edits of mCPT), 8 sessions | 1 edit accepted on RESERVED (+0.21), then −2.31 on HELDOUT (CI [−4.83, −0.68]) → reverted | sr/report.md |
| Step 8 article-frame prefix | not completed (job lost when the session ended) | archive/runs/queue3.sh |
| Second seeds (XLM-R large, silver DeBERTa e8) | not run | archive/runs/queue4.sh |

---

## 3. Diagnosis of the best CV system (mCPT + XLM-R large + silver DeBERTa), `runs/diagnosis.md`

- Ranking is good, decisions are not: per-frame AUC 0.85–0.92 (Morality 0.74, Security 0.77, Political 0.80).
- Totals: TP 593, FP 934, FN 726 → precision 0.39, recall 0.45.
- Oracles (cheating on purpose): threshold tuned on test fold 42.3; per-label thresholds on test fold 45.1;
  **perfect None decision 57.0**; **known label count + top-k 59.5**; perfect None + top-1 51.0;
  article's gold frame set known 47.7; predicting all article frames on every paragraph 33.9.
- Biggest error sources (micro gain if fixed): None +10.2 (176 None paragraphs given frames),
  Security_and_defense +6.4 (177 FP), Political +6.0. Morality almost never predicted (F1 0.09).
- Top confusions: Security instead of Political (23), Policy instead of Political (17), External_regulation instead of Political (15).
- By type: titles micro 36.2 (gold None 56%, predicted None 40%); body 42.1; long paragraphs (≥80 words) 46.3.
- Adjacent paragraphs share few labels (mean Jaccard 0.19); articles use 5.0 frames on average.

---

## 4. Codabench dev results (EN, official scorer)

| ID | submission | members (weight) | dev micro |
|---|---|---|---|
| – | phase1b_M_E1_extraEN | mCPT (M_E1) | 37.70 |
| 429 | EN_subtask1_submission (user's EN_best_model) | mpnet-base-v2 + contrastive recipe | ~41 |
| 438 | blend80_20 | user 0.8, silver DeBERTa 0.2 | 42.92 |
| 433 | blend3_60_25_15mcpt | user 0.6, DeBERTa 0.25, mCPT 0.15 | 44.08 |
| 432 | blend70_30 | user 0.7, DeBERTa 0.3 | 44.44 |
| 435/439 | blend60_40 | user 0.6, DeBERTa 0.4 | 44.54 |
| 441 | blend3_40_30_30xl | user 0.4, DeBERTa 0.3, XLM-R L 0.3 | 44.33 |
| 443 | blend3_45_35_20xl | user 0.45, DeBERTa 0.35, XLM-R L 0.2 | 44.50 |
| 442 | blend3_50_30_20xl | user 0.5, DeBERTa 0.3, XLM-R L 0.2 | 45.04 |
| **444** | **blend3_45_35_20xl_matched** | user 0.45, DeBERTa 0.35, XLM-R L 0.2, thr 1.11 (label rate matched to 60/40) | **45.07** |
| 450 | ft3_blend3_matched | as 444 with fine-tuned copy (ft1 best_e3) | 44.37 |
| 449 | ft3_blend60_40 | ft1 best_e3 0.6, DeBERTa 0.4 | 43.07 |
| **451/453** | **best45_titlerule3x** | 444 + title rule (titles need 3× threshold) | **46.90** |
| 452 | best45_titles_none | 444 + every title None | 46.59 |

Blend recipe: each model's probabilities are divided by its own global threshold (user model 0.3 from its
meta.json; DeBERTa 0.19 and XLM-R large 0.17 from our EN CV), weighted average, threshold ≈1.
Caveat: 12+ dev uploads were compared, so the best dev score is optimistic for the test set.

---

## 5. The user's model (EN_best_model) and the fine-tuned copy

- Bundle: `ckpt/EN_best_model/` (encoder = sentence-transformers/paraphrase-multilingual-mpnet-base-v2 architecture
  XLMRobertaModel 768-d, mean pooling, head Linear 768→256 · ReLU · Dropout 0.5 · Linear 256→14, threshold 0.3,
  max_len 512; label-aware contrastive training, SemEval-23 data for pre-training, selected on 2b gold).
- Reproduced exactly: 558/558 dev predictions identical (`src/ext_model.py`). Checksums in `ckpt/EN_best_model.sha256`.
- Fine-tuned COPY (`src/ext_finetune.py`, run ft1): train = EN gold replay (CV EN + batch-1 EN) + 3,454 translations
  + 2b BG/PT gold; val = 224 held-out 2b EN; lr 1e-5 / head 1e-4, bs 8×2, 8 epochs, bf16, grad ckpt.
  Held-out 2b EN micro by epoch: 46.76 (original), 46.20, **47.34 (saved)**, **48.04 (saved)**, 45.90, 46.70, 45.22, 44.90, 43.91.
  Saved: `ckpt/EN_best_model_ft/ft1/best_e2/`, `best_e3/`. On Codabench dev it did not transfer (44.37 vs 45.07).
- Title rule on the original user model, held-out 2b EN: 46.76 → 48.38 (multiplier 2.5–3.0).

---

## 6. Negative results and lessons (facts only)

1. Per-label thresholds overfit on ~900 EN paragraphs; one global threshold is better for every model.
2. Silver pretraining helps a model without framing knowledge (DeBERTa +13.6) but hurts mCPT (−1.2).
3. Contrastive objectives and translated data changed mCPT alone by −1.4 … +0.7; averaging variants +2.3 (optimistic).
4. Most of the dev gain came from the first added model (+3.5 with DeBERTa); XLM-R large added +0.5.
5. None / label-count decisions are the main error source; the scores do not predict the label count (39% accuracy).
6. Gains measured on 224 held-out paragraphs (+1.28) did not transfer to dev (−0.7): small validation sets are noisy.
7. Title paragraphs are mostly None; a stricter title threshold helps in CV (+1.0) and on 2b (+1.6).

---

## 7. Reproduce (commands, from the project root, `.venv`)

| what | command |
|---|---|
| CV run (any model) | `python -m src.train_cv --phase <dir> --name <run> --kind finetune --model {mcpt,xlmr-base,xlmr-large,mpnet} [--ctx] [--extra en] [--trans data/trans/trans_en.parquet] [--en-only] [--supcon 0.5] [--supcon-pre 3] [--init-enc ckpt/...]` |
| DeBERTa CV | `python -m src.deberta_cv --name deb_silver_bce --loss bce --init ckpt/deb_silver.pt` |
| silver data / pretraining | `python -m src.silver`; `python -m src.silver_pretrain` (DeBERTa); `python -m src.mcpt_silver` (mCPT) |
| translations | `python -m src.translate` |
| score mCPT-type runs on EN | `python -m src.score_mcpt phase2m/<run>` |
| model selection over all models | `python -m src.step10 [--write]` |
| diagnosis | `python -m src.diagnose` → runs/diagnosis.md |
| final retrain on all data (dev probs) | `python -m src.final_model --run phase1c/XL_E1 --lang en` |
| user model dev probs | `python -m src.ext_model` |
| fine-tune a copy of the user model | `python -m src.ext_finetune --run ft1 --epochs 8` (track: `bash runs/track_ft.sh`) |
| blends | `python -m src.blend_fixed`, `src.blend_variants`, `src.blend_xl`, `src.blend_ft`, `src.blend_title` |
| format check + zip | `python -m src.make_zip runs/submissions/<file>.txt en` |
| progress trackers | `bash runs/track.sh` (CV runs), `bash runs/track_ft.sh` (fine-tune) |

Hardware: one RTX 4060 8 GB. Typical times: mCPT CV 26–39 min; XLM-R large CV 89 min; DeBERTa CV ~30 min (silver
init ~45); silver pretraining DeBERTa 99 min, mCPT 5 min; NLLB translation 3.7 min; fine-tune copy 1.5 min/epoch.

---

## 8. mCPT re-implementation notebook (`notebooks/mcpt_semeval2027_task9.ipynb`)

Colab notebook that re-implements mCPT and ran the EN hyperparameter search that produced EN_best_model. Full
hyperparameters, data table, unit tests and per-language results are in README Section 4. Key numbers
(cross-fitted micro-F1 on gold 2b dev paragraphs):

| run | BG | EN | PT | RU | pooled |
|---|---|---|---|---|---|
| phase 1 (pooled multilingual, contrastive epoch 5) | .515 | .364 | .470 | .535 | .452 |
| phase 1 final (best epoch) | – | .401 | – | – | .476 |
| E0 replication (EN few-shot) | | .399 | | | |
| E1 + all 2027 languages in phase 2 | | .380 | | | |
| E2 mpnet-base-v2, no phase 1 | | .453 | | | |
| EN_best_model (best E2 epoch, chosen on the same dev) | | .461 (micro .468) | | | |

Codabench dev for EN_best_model alone: ~41 (selection on the 224 dev paragraphs made .461 optimistic).

## 9. Latest dev results (2026-11-26 20:24)

- best45_titlerule3x (45.07 blend + titles need 3× threshold): **46.90** (IDs 451, 453).
- best45_titles_none (every title None): 46.59 (ID 452).

## 10. Final analyses (src/analysis_final.py -> runs/analysis_final.md)

CV ablation, paired article-bootstrap CIs (every ensemble step and the title rule exclude 0; title rule +1.07 [+0.22, +1.97]),
per-frame F1 with/without the title rule (None F1 0.31 -> 0.41; frames within +-0.01), EN_best_model on held-out 2b
(46.76 -> 48.38). Copied into README Section 5.
