#!/usr/bin/env bash
# mCPT paper method on ENGLISH text (EN gold + extra EN + 3454 EN translations), from ORIGINAL mCPT weights.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M="--kind finetune --model mcpt --ctx --lr 5e-5 --head-lr 1e-3 --epochs 12 --bs 32 --extra en --trans data/trans/trans_en.parquet --en-only"
T() { n=$1; shift; echo "=== $n $(date +%T)"; .venv/bin/python -m src.train_cv --phase phase2m --name $n $M "$@" > runs/phase2m/$n.log 2>&1 && tail -1 runs/phase2m/$n.log || echo "FAILED $n"; }
T C1_joint --supcon 0.5
T C2_pre_then_bce --supcon-pre 3
T C3_pre_joint --supcon-pre 3 --supcon 0.5
echo "=== MCPT QUEUE DONE $(date +%T)"
