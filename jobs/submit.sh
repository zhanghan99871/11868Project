#!/usr/bin/env bash
# Interact-style allocation (salloc -J Interact) under nohup, so it survives the client disconnecting.
# usage: jobs/submit.sh <walltime> <logname> <script.sh> [args...]
T=$1; NAME=$2; shift 2
LOG=/jet/home/yni3/11868/project/logs/$NAME.log
nohup salloc -J Interact -p GPU-shared --gres=gpu:v100-32:1 -n 1 -c 5 -t "$T" -A cis260267p \
  srun -n 1 -c 5 bash "$@" > "$LOG" 2>&1 &
echo "salloc pid $! -> $LOG"
