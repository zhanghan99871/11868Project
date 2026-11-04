#!/usr/bin/env bash
# Stage 3b: greedy sweep (part 2), sampling, and the no-resident ablation.
P=/jet/home/yni3/11868/project
bash $P/jobs/run.sh specdec.profile_spec
bash $P/jobs/run.sh specdec.bench --configs spec_k4 spec_k6 spec_k3_t1.0 \
  --n-prompts 10 --n-new 64 --out $P/results/main.jsonl
bash $P/jobs/run.sh specdec.bench --no-resident --configs micro ar_target spec_k3 \
  --n-prompts 3 --n-new 32 --out $P/results/ablation_noresident.jsonl
echo STAGE3B_DONE
