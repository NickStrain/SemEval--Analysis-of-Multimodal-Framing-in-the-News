# SemEval-2027 Task 9 — Analysis of Multimodal Framing in the News: Subtask 1 (English)

Code for our system for **Subtask 1: paragraph-level framing detection**. Each paragraph of a news article is
labelled with any subset of 14 policy frames (Economic, Capacity_and_resources, Morality, Fairness_and_equality,
Legality_Constitutionality_and_jurisprudence, Policy_prescription_and_evaluation, Crime_and_punishment,
Security_and_defense, Health_and_safety, Quality_of_life, Cultural_identity, Public_opinion, Political,
External_regulation_and_reputation), or with none of them ("None"). Training data covers Bulgarian, English,
Portuguese and Russian; **this work targets English**.

- Official metric: micro-F1 over 15 classes (14 frames + None), `zero_division=1`.
- **Best English dev result (Codabench): 46.90 micro-F1** — a fixed-weight blend of three models plus a title rule (Section 5).
- Every experiment, number and command is recorded in [`runs/EXPERIMENT_LOG.md`](runs/EXPERIMENT_LOG.md).

---

## 1. Results at a glance

**English, 5-fold cross-validation by article** (epoch, weights and threshold for each test fold chosen on the
other four folds only):

| system | EN micro-F1 |
|---|---|
| mCPT, label prototypes, no training | 32.1 |
| mCPT fine-tuned (context + extra English data) | 36.82 ± 1.46 |
| DeBERTa-v3-large, gold data only | 23.4 – 25.0 |
| DeBERTa-v3-large, silver pre-training → gold | 37.01 ± 2.99 |
| XLM-R large | 38.97 ± 0.89 |
| mCPT + XLM-R large + silver DeBERTa (weighted, one global threshold) | 41.62 ± 1.04 |
| + stricter threshold for title paragraphs | **42.63 ± 1.39** |

**English dev set (Codabench, official scorer):**

| submission | dev micro-F1 |
|---|---|
| mCPT fine-tuned | 37.70 |
| EN_best_model (mpnet, label-aware contrastive training) | ~41 |
| EN_best_model 0.6 + silver DeBERTa 0.4 | 44.54 |
| EN_best_model 0.45 + silver DeBERTa 0.35 + XLM-R large 0.20 | 45.07 |
| **same + title rule (title paragraphs need 3× the threshold)** | **46.90** |
| same + every title predicted None | 46.59 |

Full tables (all variants, per-fold scores, negative results, diagnosis): [`runs/EXPERIMENT_LOG.md`](runs/EXPERIMENT_LOG.md).

### Other languages (5-fold CV, our pipeline, official micro-F1 ×100)

All models are trained on the four languages together; English is the hardest language for every model.

| model | all | EN | BG | PT | RU |
|---|---|---|---|---|---|
| mCPT fine-tuned, E0 (task data only) | 51.8 ± 0.9 | 35.6 ± 3.8 | 44.6 ± 4.1 | 62.0 ± 3.6 | 60.1 ± 3.2 |
| mCPT fine-tuned, E1 (+ batch-1 English) | 52.4 ± 1.2 | 36.8 ± 2.4 | 43.4 ± 4.2 | 63.7 ± 3.3 | 58.7 ± 4.0 |
| mCPT fine-tuned, E3 (+ batch-1 English, English ×2) | 51.9 ± 1.2 | 36.2 ± 2.5 | 44.0 ± 4.9 | 62.4 ± 3.4 | 58.3 ± 3.1 |
| XLM-R base, E0 | 52.8 ± 0.7 | 35.5 ± 3.7 | 43.8 ± 6.6 | 64.3 ± 2.3 | 60.5 ± 2.7 |

(source: `runs/phase1b/report.md`; global thresholds, implicit None, cross-fitted)

See Section 4 for the notebook's per-language results.

---

## 2. Setup

Tested with Python 3.12, PyTorch 2.11 (CUDA 12.8), transformers 5.24, scikit-learn 1.9, LightGBM 4.7, on one
RTX 4060 (8 GB).

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

All commands below are run from the repository root as modules: `python -m src.<script>`.

### Data (not included — shared-task data)

Place the task releases under `data/` with these names:

```
data/
  1_initial-training-data/        # batch 1 (SEMEVAL23_FRAMING_INITIAL_TRAIN-A/B)
  2_training-second-batch-oct-revised/{BG,EN,PT}/
  3_dev-set-nov-release/{BG,EN,PT}/   # subtask-1-participant.txt + raw-documents/
  4_training-russian-nov/RU/
  splits.json                     # included: locked final split + 5 CV folds (by article)
  folds.json                      # included: English folds
  silver/framing_subset.csv       # copenlu/mm-framing (Hugging Face), MIT licence
```

The official scorer comes from the starter kit (not included): place it at `starter-kit/starter-kit-v2/subtask1/scorer-subtask-1.py`.
Models are downloaded from Hugging Face on first use (`socialcomplab/mcpt-body-semeval2023task3`,
`FacebookAI/xlm-roberta-large`, `microsoft/deberta-v3-large`, `facebook/nllb-200-distilled-600M`,
`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`). Checkpoints (`ckpt/`) are not in git (8 GB).

---

## 3. Repository layout

```
src/                 all code (one module per step, run with python -m src.<name>)
notebooks/           mCPT re-implementation + EN hyperparameter search (Colab), produced EN_best_model
sr/                  self-rewriting experiment (gradient-free weight edits of mCPT)
runs/                logs, CV out-of-fold scores, reports, submissions
  EXPERIMENT_LOG.md  complete experiment record (read this first)
  log.csv            one row per result
  diagnosis.md       error analysis of the best CV system
  phase1*/, phase2m/, step4/   per-run folders: config.json, oof.npz / scores.npz, train_log.jsonl
  submissions/       dev prediction files (.txt) and Codabench zips
configs/             final blend weights/thresholds, silver label mapping, teacher prompt
oof/, dev/           saved out-of-fold / dev probabilities of early models
teacher/             LLM-teacher pilot inputs and labels
ckpt/                model checkpoints (git-ignored)
```

### Main modules

| module | purpose |
|---|---|
| `data.py` | loads all sources into one paragraph table (title, previous/next paragraph, 14-dim labels); locked final split |
| `eval.py`, `score.py` | official Subtask-1 scoring (verified identical to the starter-kit scorer by `verify_eval.py`) |
| `train_cv.py` | 5-fold CV training for mCPT / XLM-R / mpnet; options for context, extra data, translations, English-only training, label-aware contrastive loss (`--supcon`, `--supcon-pre`), encoder init |
| `deberta_cv.py` | 5-fold CV for DeBERTa-v3-large (English), optional silver initialisation |
| `silver.py`, `silver_pretrain.py`, `mcpt_silver.py` | build silver data from mm-framing; silver pre-training for DeBERTa / mCPT |
| `translate.py` | NLLB translation of BG/PT/RU gold paragraphs into English (3,454 rows) |
| `decode.py`, `post.py`, `ens.py` | thresholds, None handling, nested ensembling |
| `step10.py` | nested model selection over every model + submission writer |
| `score_mcpt.py`, `mcpt_avg_ens.py` | English scoring of mCPT variants, alone and in the ensemble |
| `diagnose.py`, `step3.py`, `title_rule.py` | error analysis, oracle upper bounds, title-paragraph rule |
| `stack.py` | LightGBM co-occurrence stacker (negative result) |
| `teacher.py` | LLM-teacher pilot (260 English paragraphs) |
| `final_model.py` | retrain a CV configuration on all training data and predict the dev set |
| `ext_model.py`, `ext_finetune.py` | load the EN_best_model bundle / fine-tune a copy of it (original never modified) |
| `blend_*.py` | fixed-weight dev blends used for the Codabench submissions |
| `make_zip.py` | official format check + Codabench zip |

---

## 4. mCPT re-implementation notebook (`notebooks/mcpt_semeval2027_task9.ipynb`)

A self-contained Colab notebook that re-implements mCPT (Reiter-Haas, Ertl, Innerebner & Lex, SemEval-2023 Task 3;
[paper](https://aclanthology.org/2023.semeval-1.130/), [reference code](https://github.com/socialcomplab/semeval23-mcpt))
for paragraph-level framing, runs the hyperparameter search for English (the "EN push"), and produced the
**EN_best_model** used in the best submission. Choices not stated in the paper are tagged `ASSUMPTION` in the code;
places where the paper and the reference code disagree are tagged `REF-DIFF`.

**Pipeline.** Encoder + mean pooling (no L2 normalisation) + head Linear–ReLU–Dropout–Linear (paper variant).
Loss = BCE + α · label-aware contrastive loss (HeroCon-style, paper Eq. 2–3).
- Phase 1 (pooled multilingual training: 2027 train BG/EN/PT/RU + SemEval-2023 data in DE, FR, IT, PL, RU, EN, ES, GR, KA):
  head pre-training with the encoder frozen (10 epochs), then contrastive end-to-end training (up to 50 epochs).
- Phase 2 (few-shot, per target language): head pre-training (10), contrastive stage (50), head post-training (50).
- Epoch selection on an article-grouped 15% slice of the target language's *train* data, early stopping with patience 5.
- Threshold: one global threshold on a 0.05 grid; reported scores are cross-fitted (`cf_micro_f1`: threshold tuned on
  half of the dev articles, scored on the other half).

**Hyperparameters (Config dataclass, cell 6).**

| group | setting |
|---|---|
| encoder | `paraphrase-multilingual-MiniLM-L12-v2` (384-d); EN push E2: `paraphrase-multilingual-mpnet-base-v2` (768-d), max length 512 |
| head | hidden 256, dropout 0.5, `head_variant="paper"` |
| contrastive loss | α = 0.01, temperature 1.0, class reduction mean, classes with < 2 positives skipped, no self-pairs |
| optimisation | batch 26, head lr 1e-3, encoder lr 2e-5, exponential decay per epoch (head γ 0.99, encoder γ 0.98), random sampler (contrast sampler optional) |
| schedule | phase 1: 10 head + 50 contrastive epochs; phase 2: 10 head + 50 contrastive + 50 head post-training |
| data | dev = gold 2b paragraphs (EN, BG, PT), RU = 20% article-grouped split; SemEval-2023 data in phase 1 only |
| leakage control | duplicate-aware article split (texts ≥ 30 chars shared by ≤ 5 articles), dev paragraphs that (near-)duplicate any training text are dropped (TF-IDF char 3–5-gram cosine ≥ 0.95) |
| seed | 42 (EN push ensemble: 42, 43, 44) |

Nine unit tests (cell 30) pass: CRLF offsets and empty labels, duplicate articles kept in one split, label
normalisation, loss finite and equal to the reference, export format round-trip, no split leakage, contrast sampler
covers all classes, finite contrastive loss in training batches, encoder frozen then trained.

**Data seen by the notebook (cell 17, after the leakage filter).**

| set | paragraphs | articles | empty (None) | mean #labels |
|---|---|---|---|---|
| BG train / dev | 881 / 180 | 100 / 20 | 306 / 40 | 1.04 / 1.62 |
| EN train / dev | 1036 / 224 | 96 / 20 | 258 / 66 | 1.20 / 1.16 |
| PT train / dev | 1298 / 240 | 105 / 20 | 104 / 35 | 1.76 / 1.45 |
| RU train / dev | 367 / 78 | 37 / 10 | 33 / 9 | 1.96 / 1.97 |
| SemEval-2023 / batch-1 extra (9 languages) | 5,876 | 1,547 | 0 | 1.14–1.44 |
| pooled phase-1 train | 9,458 | 1,885 | 701 | 1.36 |

Majority-label baselines on dev (micro-F1): BG .156, EN .240, PT .112, RU .266.

**Phase 1, per-language dev results (pooled multilingual model, cross-fitted micro-F1).**

| checkpoint | BG | EN | PT | RU | pooled |
|---|---|---|---|---|---|
| head pre-training, epoch 5 | .473 | .358 | .454 | .492 | .412 |
| head pre-training, epoch 10 | .486 | .359 | .481 | .521 | .440 |
| contrastive, epoch 5 | .515 | .364 | .470 | .535 | .452 |
| final phase-1 model (best epoch restored) | – | .401 | – | – | .476 (micro .485, macro .432, thr 0.25) |

**EN push (English dev = 224 gold 2b paragraphs; cross-fitted micro-F1).**

| id | change | cf micro | micro | macro | threshold |
|---|---|---|---|---|---|
| baseline | TF-IDF + logistic regression | ≈ .36 | | | |
| E0 | replication: phase 1 + EN-only few-shot | .399 | .399 | .320 | 0.35 |
| E1 | E0 + phase 2 on all four 2027 languages + best-epoch selection | .380 | .408 | .356 | 0.30 |
| **E2** | E1 with mpnet-base-v2, no phase 1 | **.453** | .454 | .380 | 0.35 |
| saved EN_best_model | best E2 epoch (head post-training, epoch 6) | .461 | .468 | .397 | 0.30 |

Also tried and dropped: per-label thresholds and the previous paragraph as context (both lowered the TF-IDF
baseline). The 3-seed ensemble (ENS) and the ablations (−PT, −L_CON, −E2E, +CS) were started; their final
tables are not in the saved outputs (the −L_CON run reached .446 at head-pretraining epoch 6 before stopping).

Caveat (recorded in the bundle's `meta.json`): the saved model was chosen as the maximum over many evaluations
on the same 224 dev paragraphs, so its .461–.468 is optimistic; on the Codabench dev set it scores ~41.

**Running it:** upload the release zip to Colab, set `SMOKE_TEST=False` (True runs a 1–2 min synthetic end-to-end
check), run all. Do **not** put an access token in the notebook (public models need none).

---

## 5. Reproducing the main results

**Cross-validation runs** (each writes `runs/<phase>/<name>/oof.npz`):

```bash
# mCPT fine-tuned (context, extra English data). Note: --extra en now also loads the 224 dev-release 2b EN gold
# paragraphs (added after this run was made), so a re-run trains on slightly more data than the logged M_E1.
python -m src.train_cv --phase phase1b --name M_E1_extraEN --kind finetune --model mcpt --ctx --lr 5e-5 --epochs 12 --extra en
# XLM-R large
python -m src.train_cv --phase phase1c --name XL_E1 --kind finetune --model xlmr-large --lr 1e-5 --epochs 8 --freeze-emb --grad-ckpt --extra en
# DeBERTa-v3-large with silver pre-training
python -m src.silver && python -m src.silver_pretrain
python -m src.deberta_cv --name deb_silver_bce --loss bce --init ckpt/deb_silver.pt
# mCPT variants (translations, contrastive loss)
python -m src.translate
python -m src.train_cv --phase phase2m --name C1_joint --kind finetune --model mcpt --ctx --lr 5e-5 --epochs 12 --bs 32 \
       --extra en --trans data/trans/trans_en.parquet --en-only --supcon 0.5
```

**Scoring and analysis:**

```bash
python -m src.step10              # nested selection over all models -> runs/step10.json
python -m src.score_mcpt phase2m/C1_joint
python -m src.diagnose            # -> runs/diagnosis.md
python -m src.title_rule
```

**Dev submissions (45.07, then 46.90 with the title rule):**

```bash
python -m src.final_model --run phase1c/XL_E1 --lang en   # XLM-R large dev probabilities
python -m src.ext_model                                    # EN_best_model dev probabilities (bundle in ckpt/EN_best_model)
python -m src.blend_xl                                     # writes blend3_45_35_20xl_matched_EN_dev.txt
python -m src.make_zip runs/submissions/blend3_45_35_20xl_matched_EN_dev.txt en
python -m src.title_rule                                  # checks the title rule on CV and held-out 2b
python -m src.blend_title                                  # 45.07 blend + title rule -> best45_titlerule3x_EN_dev.txt
python -m src.make_zip runs/submissions/best45_titlerule3x_EN_dev.txt en
```

Blend rule: each model's probabilities are divided by its own global threshold (EN_best_model 0.30, silver
DeBERTa 0.19 and XLM-R large 0.17, the latter two tuned on English CV), averaged with weights 0.45 / 0.35 / 0.20,
and labels with blended score ≥ 1.11 are predicted. Title rule: for the first paragraph of each article (the title)
the threshold is 3 × 1.11; the factor was chosen on our English CV (all five folds chose 3.0) and confirmed on the held-out
2b paragraphs (46.76 → 48.38), not on dev.

**Progress while training:** `bash runs/track.sh` (CV runs), `bash runs/track_ft.sh` (fine-tuning).

Typical run times on the RTX 4060: mCPT CV 26–39 min, XLM-R large CV 89 min, DeBERTa CV 30–45 min,
DeBERTa silver pre-training 99 min, translation 4 min.

---

## 6. Evaluation protocol (validity)

- **Folds:** 5 folds split by article (no article in two folds), stratified by language; scores are on English rows.
- **No test-fold tuning:** for each test fold, the epoch, ensemble weights and thresholds are chosen on the other
  four folds only ("nested"). Oracle numbers in the diagnosis are labelled as such.
- **Locked split:** 35 articles held out from all model selection (`data/splits.json`).
- **Translations:** a translated paragraph inherits its source article's fold and is never used to train the fold
  that contains its source article.
- **Dev:** dev labels were never available; dev numbers come only from Codabench uploads. Several uploads were
  compared, so the best dev score is optimistic for the test set.

---

## 7. Licence and data

Code: see `LICENSE`. Shared-task data, the mm-framing corpus and pretrained models keep their own licences and are
not redistributed here.
