#!/usr/bin/env bash
# Resume after the 01:35 kill: DeBERTa deb_bce fold 4 -> queue2 -> queue3 (sequential, detached).
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== fold 4 deb_bce (resume) $(date +%T)" >> runs/step4/queue.log
.venv/bin/python -m src.deberta_cv --name deb_bce --loss bce --folds 4 2>&1 | grep -E '^\{|Error|Traceback' >> runs/step4/queue.log
echo "=== STEP4 DONE $(date +%T)" >> runs/step4/queue.log
bash runs/queue2.sh >> runs/queue2.log 2>&1
bash runs/queue3.sh >> runs/queue3.log 2>&1
echo "=== CHAIN DONE $(date +%T)" >> runs/queue3.log
