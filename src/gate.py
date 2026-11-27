"""Gate rule for later weight edits (Phase 2+).

An edit PASSES iff, on the 5 CV folds, against the unedited model with the SAME decoding:
  1. mean (over folds) official macro-F1 improves by >= 0.5 points (0.005), AND
  2. macro-F1 is strictly higher in >= 4 of the 5 folds.
macro-F1 = official 15-class macro-F1 (None included, zero_division=1) on each fold's
pooled test paragraphs (all languages).
"""
import numpy as np

MIN_MEAN_GAIN = 0.005
MIN_FOLDS_BETTER = 4


def gate(base_macro_per_fold, edit_macro_per_fold) -> dict:
    b, e = np.asarray(base_macro_per_fold, float), np.asarray(edit_macro_per_fold, float)
    gain = float(e.mean() - b.mean())
    better = int((e > b).sum())
    return dict(mean_gain=gain, folds_better=better,
                passed=gain >= MIN_MEAN_GAIN and better >= MIN_FOLDS_BETTER)
