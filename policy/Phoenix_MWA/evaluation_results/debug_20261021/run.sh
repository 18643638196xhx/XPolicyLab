#!/bin/bash
set -euo pipefail
root=/mnt/data/wujie/phoenix_robotwin_debug_20261021
export PATH=/opt/conda/envs/fastwam-pretrain/bin:/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=/opt/conda/envs/fastwam-pretrain/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$root:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export EVAL_ENV_TYPE=debug
cd "$root/XPolicyLab/policy/Phoenix_MWA"
for mode in plain encoded; do
  if [ "$mode" = encoded ]; then export DEBUG_OBS_ENCODED=1; else export DEBUG_OBS_ENCODED=0; fi
  echo "RUNNING $mode" > "$root/status.txt"
  set +e
  bash eval.sh RoboTwin adjust_bottle phoenix-robotwin-35k arx_x5 joint 0 0 0 fastwam-pretrain fastwam-pretrain > "$root/$mode.log" 2>&1
  rc=$?
  set -e
  echo "$rc" > "$root/$mode.exit"
  if [ "$rc" -ne 0 ]; then echo "FAILED $mode rc=$rc" > "$root/status.txt"; exit "$rc"; fi
  mkdir -p "$root/${mode}_audit"
  if [ -d audit ]; then cp -a audit/. "$root/${mode}_audit/"; fi
  echo "PASSED $mode" > "$root/status.txt"
done
echo COMPLETE > "$root/status.txt"
