#!/usr/bin/env bash
# Runs stage3a then stage3b in back-to-back interact allocations (QOS allows one job at a time).
# Launched under nohup on the login node so it survives the client disconnecting.
A="-J Interact -p GPU-shared --gres=gpu:v100-32:1 -n 1 -c 5 -A cis260267p"
L=/jet/home/yni3/11868/project/logs
salloc $A -t 00:58:00 srun -n 1 -c 5 bash /jet/home/yni3/11868/project/jobs/stage3a.sh > $L/stage3a.log 2>&1
grep -q STAGE3A_DONE $L/stage3a.log || { echo "stage3a did not finish" >> $L/stage3a.log; exit 1; }
salloc $A -t 00:58:00 srun -n 1 -c 5 bash /jet/home/yni3/11868/project/jobs/stage3b.sh > $L/stage3b.log 2>&1
