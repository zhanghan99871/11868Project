#!/usr/bin/env bash
# Stage 1: correctness gates, then latency micro-benchmark and a 3-prompt calibration run.
P=/jet/home/yni3/11868/project
bash $P/jobs/run.sh specdec.test_correctness || { echo "CORRECTNESS FAILED, stopping"; exit 1; }
bash $P/jobs/run.sh specdec.bench --configs micro ar_target ar_draft spec_k4 --n-prompts 3 --n-new 32 \
  --out $P/results/stage1_calib.jsonl
echo STAGE1_DONE
