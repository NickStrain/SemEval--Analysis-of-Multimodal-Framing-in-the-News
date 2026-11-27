#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "runs/step4_queue.sh" >/dev/null; do sleep 30; done
T() { name=$1; shift; echo "=== $name $(date +%T)"; .venv/bin/python -m src.train_cv --phase phase1c --name "$name" --kind finetune "$@" > runs/phase1c/$name.log 2>&1 && tail -1 runs/phase1c/$name.log || echo "FAILED $name"; }
mkdir -p runs/phase1c
T XL_E1 --model xlmr-large --lr 1e-5 --epochs 8 --freeze-emb --grad-ckpt --extra en
echo "=== silver pretrain $(date +%T)"; .venv/bin/python -m src.silver_pretrain > runs/step4/silver_pretrain.log 2>&1; tail -1 runs/step4/silver_pretrain.log
W=$(cat runs/step4/winner.txt); L=${W#deb_}
echo "=== deberta gold from silver ($L) $(date +%T)"; .venv/bin/python -m src.deberta_cv --name deb_silver_$L --loss $L --init ckpt/deb_silver.pt 2>&1 | grep -E '^\{|Error' | tail -2
T XB_E1 --model xlmr-base --lr 2e-5 --epochs 10 --extra en
echo "=== finals $(date +%T)"
for r in phase1c/XL_E1 phase1/C_xlmr_base phase1c/XB_E1; do .venv/bin/python -m src.final_model --run $r --lang en > runs/final_$(echo $r | tr / _).log 2>&1; tail -1 runs/final_$(echo $r | tr / _).log; done
echo "=== QUEUE2 DONE $(date +%T)"
