#!/usr/bin/env bash
# Phase 1 training queue (sequential, one GPU job at a time).
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { name=$1; shift; echo "=== $name $(date +%T)"; .venv/bin/python -m src.train_cv --name "$name" --kind finetune "$@" > "runs/phase1/$name.log" 2>&1 && tail -1 "runs/phase1/$name.log" || echo "FAILED $name (see log)"; }
run B_tgt_lr5e-5  --model mcpt --lr 5e-5 --epochs 12
run B_tgt_lr3e-5  --model mcpt --lr 3e-5 --epochs 12
run C_xlmr_base   --model xlmr-base --lr 2e-5 --epochs 10
run B_ctx_lr5e-5  --model mcpt --lr 5e-5 --epochs 12 --ctx
run B_ctx_lr3e-5  --model mcpt --lr 3e-5 --epochs 12 --ctx
run C2_xlmr_large --model xlmr-large --lr 1e-5 --epochs 8 --freeze-emb --grad-ckpt
echo "=== ALL DONE $(date +%T)"
