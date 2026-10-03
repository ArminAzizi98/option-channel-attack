#!/bin/bash
# Option-channel on the public screening tasks. von and rlcd are the label-invariant controls.
cd "$(dirname "$0")/.."          # repository root
# activate your environment first, e.g. source venv/bin/activate
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONPATH=src HF_HOME=hf
for m in laya-td laya-ml von rlcd; do
  echo "######## $m $(date +%H:%M:%S)"
  python src/run_optsweep_real.py --model "$m" --n 300 --out runs/optsweep_real || echo "FAILED $m"
done
echo OPTSWEEP_REAL_ALL_DONE
