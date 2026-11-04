#!/usr/bin/env bash
# Runs one python module inside a GPU allocation. Launched via jobs/submit.sh.
# usage: jobs/run.sh <module> [args...]
set -uo pipefail
PROJ=/jet/home/yni3/11868/project
module load cuda/12.4.0
source /jet/home/yni3/11868/llmsys_hw4/.venv/bin/activate
export HF_HOME=/ocean/projects/cis260267p/yni3/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=5
cd "$PROJ"
echo "=== $(hostname) $(date) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv
python -m "$@"
rc=$?
echo "=== exit=$rc $(date) ==="
exit $rc
