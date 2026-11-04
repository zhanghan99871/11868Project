#!/usr/bin/env bash
# Stage 5b: full greedy sweep with the distilgpt2 draft in ONE allocation (same-node baseline).
P=/jet/home/yni3/11868/project
bash $P/jobs/run.sh specdec.bench --draft distilgpt2 \
  --configs ar_target ar_draft spec_k1 spec_k2 spec_k3 spec_k4 spec_k6 \
  --n-prompts 10 --n-new 64 --out $P/results/sweep_distil.jsonl
echo STAGE5B_DONE
