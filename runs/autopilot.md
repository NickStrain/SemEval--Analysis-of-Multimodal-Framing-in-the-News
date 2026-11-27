# Overnight autopilot (started 2026-11-26, user asleep; goal: EN micro toward 47)
User instruction: "do all the things ... get to 47". Plan = user's 10-step plan + accepted fixes 1-7:
 - KEEP: CV micro +>=1.0 AND mean gain > std of paired per-fold differences.
 - Step-2 style tuning always cross-fitted over EN folds (tune on 4, apply to 1).
 - DeBERTa-v3-large: bf16, frozen embeddings, 8-bit Adam if needed; 2 seeds if noisy.
 - Step 8 article frames from OOF predictions only.
 - No dev scoring, no Codabench uploads, no paid API calls (Step 7 limited to my 260-paragraph pilot).
 - SR Phase 2 first (uses EN final split), then final split may join Step-10 training.
Status log (append below):
- SR Phase 2 done: accepted s8 edit HURTS on heldout (-2.3 micro, CI [-4.8,-0.7]); reverted; sr/report.md written.
- Step 1 done: score.py identical to official on 3 synthetic EN files; oof/mcpt_en.npy, dev/mcpt_en.npy saved.
- User (before sleep): continue from Step 2; don't train from scratch, build on best model M_E1 (mcpt_en).
- Step 2 NOT kept: EN-tuned post -1.82 (0/5); all-lang-tuned post -0.37 (3/5); None rule/cap never chosen. Baseline decode stays.
- Step 3 done: runs/errors.md (top cost: None +10.6, Security +7.5, Political +6.5, QoL +4.3).
- Step 2 variants, EN-only thresholds, stacker (EN-only and all-lang trained on mCPT): all NOT kept (-1 to -5).
- Step 5 KEEP: mCPT(E1)+XLM-R base, weights+thresholds nested on ALL languages: EN 38.12±3.52 (+1.34, 4/5). Lesson: ensembles of multilingual models tuned on all languages help.
- Queue2 (after Step 4): XLM-R large+extraEN CV -> silver DeBERTa -> DeBERTa gold from silver -> XLM-R base+extraEN CV -> finals.
- 01:38 Session died ~01:35 killing step4 (folds 0-3 saved) + queue2/3 unstarted. Relaunched detached: runs/resume_chain.sh (fold 4 -> queue2 -> queue3).
- 01:55 Step 4 NOT kept: deb_bce nested EN 23.4-25.0 micro (under-confident, max prob <0.5, still rising at ep5; in-fold-tuned ~33). Adding it to mCPT+XLM-R does not help (37.04 vs 37.52 EN-only-tuned). Wrote src/step10.py (nested selection over all models + final writer).
- 03:20 XL_E1 (XLM-R large+extraEN) CV done: best single model 38.46±2.88 (EN nested). Best ensemble xlmrb+xlmrl+deb 38.58. Gains vs mCPT+XLM-R base (37.5) ~+1, about 1/3 of the std -> KEEP rule borderline.
- 03:30 Added --seed to train_cv/final_model (default 13, unchanged behaviour). Queued runs/queue4.sh (XL_E1 seed 101 CV + final) after the resume chain.
- 05:45 Step 6 KEEP: silver pretraining lifts DeBERTa nested 23.4 -> 34.5 (per-label thr) / 37.0 (global thr).
  Finding: ONE global threshold generalises better than per-label thresholds for every model (per-label overfits on ~700 rows).
  Best: mCPT + XLM-R large + deb_silver_bce, global thr, EN-tuned: 41.62 ± 1.04 (per-fold 42.4 39.6 42.4 42.0 41.6), macro 37.68. +4.1 over Step-5 baseline -> KEEP.
- 05:50 queue4 reordered: deb_silver_e8 (silver init, 8 ep, seed 14) first, then XL_E1 seed 101.
- 05:50 PAUSED at user request ("store all and have a small break"). Jobs frozen with SIGSTOP (pids 65146 resume_chain, 65709 queue2, 68611 after_chain, 73029 XB_E1 train_cv); nothing lost.
  RESUME: bash runs/resume.sh
  Remaining after resume: XB_E1 CV (fold 1/5) -> finals (XL_E1, C_xlmr_base, XB_E1) -> queue3 XB_E1_AF -> queue4 (deb_silver_e8, XL_E1_s2).
  Submission: once runs/submissions/phase1c_XL_E1_EN_dev_scores.npy exists -> .venv/bin/python -m src.step10 --write ; .venv/bin/python -m src.make_zip runs/submissions/final_EN_dev.txt en
  Best so far: mCPT + XLM-R large + deb_silver_bce, global thr: 41.62 ± 1.04 (snapshot runs/step10_0545_best41.6.json).
- 12:59 mCPT EN push: S1 silver-init mCPT 35.65±3.06 vs original M_E1 36.82±1.46 (EN, nested epoch+global thr) -> silver init NOT kept for mCPT. Relaunched S3 (EN-only text: EN gold + 3454 NLLB translations) and S2 (all + translations + 3 EN-only epochs) from ORIGINAL mCPT.
- 13:45 S3 (EN-only text + translations) 36.31±3.51 alone; S1+S3 avg 38.38±3.51. Stopped S2 (multilingual) on user request 'focus on en'. Next: label-aware contrastive (mCPT paper) on EN text.
- 14:22 C1 joint BCE+contrastive (EN text): 37.48±2.76 alone (+0.66 vs 36.82, below KEEP), ensemble 38.87 (vs 39.22). Not kept; C2/C3 running.
- 14:58 C2 (paper 2-stage: contrastive then BCE, EN text): 35.39±3.73 alone (-1.4), ensemble 40.16. Not kept.
- 15:35 C3 (contrastive pre + joint): 35.30±2.30 alone, ensemble 40.31. Not kept. mCPT queue done.
- 19:17 User's EN_best_model (mpnet-base-v2 + contrastive recipe, Codabench dev 41): reproduced 100%, dev probs saved runs/submissions/EN_best_model_EN_dev_scores.npy. Running X1_mpnet in our CV for honest OOF.
- 19:23 X1_mpnet CV stopped at user request (no OOF).
- 19:27 Codabench dev: EN_best_model 41 -> blend70_30 (with silver DeBERTa) 44.4.
- 19:33 Codabench dev: 80/20 42.92, 70/30 44.44, 60/40 44.54 (best), +mCPT 44.08. Weight curve flat at 60-70% -> stop tuning weights on dev.
- 20:19 ft1 (copy of EN_best_model, +translations +2b BG/PT, replay EN): held-out 2b-EN 46.76 -> 48.04 at epoch 3 (saved ckpt/EN_best_model_ft/ft1/best_e3); overfits after. Made ft3 blends.
- 20:21 Codabench dev: ft3_blend3_matched 44.37 (< 45.07), ft3_blend60_40 43.07 (< 44.54): fine-tuned copy's +1.28 on 2b-EN did NOT transfer. Dropped.
- 20:22 Title rule (titles need 3x threshold): CV EN 41.62 -> 42.63 (nested, 4/5 folds up), user model on 2b-EN 46.76 -> 48.38. Applied to the 45.07 blend -> best45_titlerule3x / best45_titles_none.
