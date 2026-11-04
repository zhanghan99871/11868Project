#!/usr/bin/env bash
# Stage 5a: correctness gates with distilgpt2 as the draft, then its latency micro-benchmark.
P=/jet/home/yni3/11868/project
bash $P/jobs/run.sh specdec.test_correctness --draft distilgpt2 || { echo "CORRECTNESS FAILED, stopping"; exit 1; }
bash $P/jobs/run.sh specdec.bench --draft distilgpt2 --configs micro --n-prompts 1 --out $P/results/distil_micro.jsonl
echo STAGE5A_DONE
