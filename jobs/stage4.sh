#!/usr/bin/env bash
# Stage 4: full greedy sweep inside ONE allocation, so every speedup is measured against an
# ar_target baseline from the same node (stage3 split configs across v013 and v007).
P=/jet/home/yni3/11868/project
bash $P/jobs/run.sh specdec.bench --configs ar_target spec_k1 spec_k2 spec_k3 spec_k4 spec_k6 \
  --n-prompts 10 --n-new 64 --out $P/results/sweep_same_node.jsonl
echo STAGE4_DONE
