#!/usr/bin/env bash
# XLM-R large final retrain (dev scores) -> best-system selection + submission + format check.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== XL_E1 final $(date +%T)"
.venv/bin/python -m src.final_model --run phase1c/XL_E1 --lang en > runs/final_phase1c_XL_E1.log 2>&1; tail -1 runs/final_phase1c_XL_E1.log
echo "=== step10 --write $(date +%T)"
.venv/bin/python -m src.step10 --write > runs/step10_write.log 2>&1; grep -E "FINAL|Error|Trace" runs/step10_write.log
.venv/bin/python -m src.make_zip runs/submissions/final_EN_dev.txt en 2>&1 | tail -2
echo "=== DONE $(date +%T)"
