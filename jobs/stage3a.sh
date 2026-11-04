#!/usr/bin/env bash
# Stage 3a: re-gate correctness on the resident-weight path, then main greedy sweep (part 1).
P=/jet/home/yni3/11868/project
bash $P/jobs/run.sh specdec.test_correctness || { echo "CORRECTNESS FAILED, stopping"; exit 1; }
bash $P/jobs/run.sh specdec.bench --configs ar_target ar_draft spec_k1 spec_k2 spec_k3 \
  --n-prompts 10 --n-new 64 --out $P/results/main.jsonl
echo STAGE3A_DONE
