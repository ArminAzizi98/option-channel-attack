#!/bin/bash
# Stage 1: full 198-label vocabulary on the five typed decision models.
# von and rlcd are the controls: they never put the label in the model's input.
cd "$(dirname "$0")/.."          # repository root
# activate your environment first, e.g. source venv/bin/activate
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}          # GPU 0 belongs to the other project
export PYTHONPATH=src
export HF_HOME=${HF_HOME:-$PWD/hf}
for m in laya-td laya-en laya-ml von rlcd; do
  echo "######## $m $(date +%H:%M:%S)"
  python src/run_optsweep.py --model "$m" --n 200 --out runs/optsweep || echo "FAILED $m"
done
echo OPTSWEEP_S1_DONE
