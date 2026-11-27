#!/usr/bin/env bash
# mpnet (encoder of the user's EN_best_model, dev 41 on Codabench) in OUR 5-fold CV -> honest OOF for ensembling.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
n=X1_mpnet; echo "=== $n $(date +%T)"
.venv/bin/python -m src.train_cv --phase phase2m --name $n --kind finetune --model mpnet --ctx --lr 2e-5 --head-lr 1e-3 --epochs 10 --bs 16 --extra en --supcon 0.5 > runs/phase2m/$n.log 2>&1 && tail -1 runs/phase2m/$n.log || echo "FAILED $n"
echo "=== MPNET DONE $(date +%T)"
