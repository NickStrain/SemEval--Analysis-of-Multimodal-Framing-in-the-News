"""Official Subtask-1 scoring (None counted as a label, 15 classes, zero_division=1).
Thin alias of src.eval (verified identical to starter-kit scorer-subtask-1.py)."""
from src.eval import official_scores  # noqa: F401  (micro, macro) from (N,14) gold and pred
