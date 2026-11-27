#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "runs/queue2.sh" >/dev/null; do sleep 30; done
echo "=== XB_E1_AF (Step 8) $(date +%T)"
.venv/bin/python -m src.train_cv --phase phase1c --name XB_E1_AF --kind finetune --model xlmr-base --lr 2e-5 --epochs 10 --extra en --artframes data/artframes.json > runs/phase1c/XB_E1_AF.log 2>&1 && tail -1 runs/phase1c/XB_E1_AF.log
echo "=== QUEUE3 DONE $(date +%T)"
