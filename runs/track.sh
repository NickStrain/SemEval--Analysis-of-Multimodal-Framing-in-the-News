#!/usr/bin/env bash
# Progress of the mCPT English runs.  Usage: watch -n 20 bash runs/track.sh
cd "$(dirname "$0")/.."
date +%T; echo "--- queue"; tail -3 runs/mcpt_queue.log
echo "--- runs (last line: fold / epoch, 5 folds x 12 epochs; supcon_epoch = contrastive stage)"
for f in runs/phase2m/C*.log runs/phase2m/X*.log; do r=$(basename $f .log)
  st=$([ -f runs/phase2m/$r/oof.npz ] && echo "FINISHED -> score: .venv/bin/python -m src.score_mcpt phase2m/$r" || grep -E '^\{' $f | tail -1 | grep -oE '"(fold|epoch|supcon_epoch)": [0-9]+' | tr '\n' ' ')
  printf "%-18s %s\n" "$r" "$st"; done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
