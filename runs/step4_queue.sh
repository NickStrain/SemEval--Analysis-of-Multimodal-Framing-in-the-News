#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D() { .venv/bin/python -m src.deberta_cv "$@" 2>&1 | grep -E '^\{|Error|Traceback'; }
echo "=== fold 0 BCE $(date +%T)"; D --name deb_bce --loss bce --folds 0
echo "=== fold 0 ASL $(date +%T)"; D --name deb_asl --loss asl --folds 0
.venv/bin/python -m src.step4_pick
W=$(cat runs/step4/winner.txt); L=${W#deb_}
echo "=== folds 1-4 $W $(date +%T)"; D --name $W --loss $L --folds 1,2,3,4
echo "=== STEP4 DONE $(date +%T)"
