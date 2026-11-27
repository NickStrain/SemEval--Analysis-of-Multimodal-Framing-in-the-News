#!/usr/bin/env bash
# Sessions 4-8, same schedule as sessions 1-3: a=0.05 at the best layer, 3 tuning tries, 1 bias edit.
cd "$(dirname "$0")/.."
H() { .venv/bin/python -m sr.harness "$@" 2>&1 | grep -vE 'Loading|Warn|warn|Token|saved: |\|Δ\|'; }
P() { .venv/bin/python -m sr.tools proj "$@"; }
session() {  # sid best_layer other_layer bias_c label
  s=$1; L=$2; L2=$3; c=$4; lab=$5
  echo "===== $s  $lab  (best layer $L)"
  E=$(P $s read1 $L 0.05); H $s code --file $E --why "Layer $L separates the $lab groups best. Project that direction out at a=0.05."
  echo "--- L$L a=0.05"; H $s generate --ids unread --which both --edit ${s}_projL${L}_a0.05 --why "Paired test on unread practice paragraphs."
  E=$(P $s read1 $L 0.15); H $s code --file $E --why "Tuning try 1: strengthen to a=0.15."
  echo "--- L$L a=0.15"; H $s generate --ids unread --which both --edit ${s}_projL${L}_a0.15 --why "Paired test of a=0.15."
  E=$(P $s read1 $L 0.3); H $s code --file $E --why "Tuning try 2: maximum strength a=0.3."
  echo "--- L$L a=0.3"; H $s generate --ids unread --which both --edit ${s}_projL${L}_a0.3 --why "Paired test of a=0.3."
  E=$(P $s read1 $L2 0.15); H $s code --file $E --why "Tuning try 3: the other top layer $L2 at a=0.15."
  echo "--- L$L2 a=0.15"; H $s generate --ids unread --which both --edit ${s}_projL${L2}_a0.15 --why "Paired test of layer $L2, a=0.15."
  E=$(.venv/bin/python -m sr.tools bias $s $c); H $s code --file $E --why "One bias edit for this session: pre-GELU FFN bias shifted by $c in every layer. The constant differs from all earlier sessions."
  echo "--- bias $c"; H $s generate --ids unread --which both --edit "${s}_bias$(printf '%+g' $c)" --why "Paired test of the bias shift."
}
session s4 10 12 -0.03 "Policy FP"
session s5 12 10 -0.01 "Security FP"
session s6 12 10 0.02 "Legality FN"
session s7 12 10 -0.04 "Crime FP"
session s8 10 12 0.03 "Political FN"
echo "===== DONE"
