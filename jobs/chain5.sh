#!/usr/bin/env bash
# stage5a then stage5b in back-to-back interact allocations (one job per user at a time).
A="-J Interact -p GPU-shared --gres=gpu:v100-32:1 -n 1 -c 5 -A cis260267p"
L=/jet/home/yni3/11868/project/logs
salloc $A -t 00:25:00 srun -n 1 -c 5 bash /jet/home/yni3/11868/project/jobs/stage5a.sh > $L/stage5a.log 2>&1
grep -q STAGE5A_DONE $L/stage5a.log || { echo "stage5a did not finish" >> $L/stage5a.log; exit 1; }
salloc $A -t 00:58:00 srun -n 1 -c 5 bash /jet/home/yni3/11868/project/jobs/stage5b.sh > $L/stage5b.log 2>&1
