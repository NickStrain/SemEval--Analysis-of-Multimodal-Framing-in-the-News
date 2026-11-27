#!/usr/bin/env bash
cd "$(dirname "$0")/.."
while pgrep -f "runs/resume_chain.sh" >/dev/null; do sleep 30; done
bash runs/queue4.sh >> runs/queue4.log 2>&1
