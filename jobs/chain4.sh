#!/usr/bin/env bash
# Waits for chain3 (pid $1) to release its allocation, then runs stage4.
while kill -0 "$1" 2>/dev/null; do sleep 30; done
sleep 20
salloc -J Interact -p GPU-shared --gres=gpu:v100-32:1 -n 1 -c 5 -A cis260267p -t 00:58:00 \
  srun -n 1 -c 5 bash /jet/home/yni3/11868/project/jobs/stage4.sh > /jet/home/yni3/11868/project/logs/stage4.log 2>&1
