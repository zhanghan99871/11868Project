#!/usr/bin/env bash
# Stage 2 (after the contiguous() fix): profile, correctness gates, latency micro-benchmark.
P=/jet/home/yni3/11868/project
bash $P/jobs/run.sh specdec.profile_step
bash $P/jobs/run.sh specdec.test_correctness || { echo "CORRECTNESS FAILED, stopping"; exit 1; }
bash $P/jobs/run.sh specdec.bench --configs micro --n-prompts 1 --out $P/results/stage2_micro.jsonl
echo STAGE2_DONE
