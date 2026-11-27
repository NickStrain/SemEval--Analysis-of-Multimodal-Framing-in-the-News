#!/usr/bin/env bash
# Phase 1b (resumed after the EN dev submission): X_E1, then E1 + batch-1 RU for both backbones.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
X="--model xlmr-base --lr 2e-5 --epochs 10"
M="--model mcpt --ctx --lr 5e-5 --epochs 12"
run() { name=$1; shift; echo "=== $name $(date +%T)"; .venv/bin/python -m src.train_cv --phase phase1b --name "$name" --kind finetune "$@" > "runs/phase1b/$name.log" 2>&1 && tail -1 "runs/phase1b/$name.log" || echo "FAILED $name (see log)"; }
run X_E1_extraEN       $X --extra en
run M_E5_extraENRU     $M --extra en,ru
run X_E5_extraENRU     $X --extra en,ru
echo "=== ALL DONE $(date +%T)"
