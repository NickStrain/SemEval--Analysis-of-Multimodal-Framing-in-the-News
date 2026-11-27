#!/usr/bin/env bash
# Appended overnight: (1) 2nd silver-init DeBERTa, 8 epochs, other seed; (2) XLM-R large 2nd seed CV + final.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== deb_silver_e8 $(date +%T)"
.venv/bin/python -m src.deberta_cv --name deb_silver_e8 --loss bce --init ckpt/deb_silver.pt --epochs 8 --seed 14 2>&1 | grep -E '^\{|Error|Traceback' | tail -2
echo "=== XL_E1_s2 $(date +%T)"
.venv/bin/python -m src.train_cv --phase phase1c --name XL_E1_s2 --kind finetune --model xlmr-large --lr 1e-5 --epochs 8 --freeze-emb --grad-ckpt --extra en --seed 101 > runs/phase1c/XL_E1_s2.log 2>&1 && tail -1 runs/phase1c/XL_E1_s2.log || echo "FAILED XL_E1_s2"
.venv/bin/python -m src.final_model --run phase1c/XL_E1_s2 --lang en > runs/final_phase1c_XL_E1_s2.log 2>&1; tail -1 runs/final_phase1c_XL_E1_s2.log
echo "=== QUEUE4 DONE $(date +%T)"
