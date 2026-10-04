#!/bin/bash
# Adaptive test: search the full 198-label vocabulary against the DEFENDED configuration.
cd "$(dirname "$0")/.."          # repository root
# activate your environment first, e.g. source venv/bin/activate
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONPATH=src HF_HOME=hf
echo "######## laya-td (control, instruction did not help it) $(date +%H:%M:%S)"
python src/run_optsweep.py --model laya-td --n 200 --instruction I1_definitions_only \
  --out runs/optsweep_defended || echo "FAILED laya-td"
echo "######## qwen7b (the model the instruction defended) $(date +%H:%M:%S)"
python src/run_optsweep.py --model qwen7b --n 100 --instruction I1_definitions_only \
  --out runs/optsweep_defended || echo "FAILED qwen7b"
echo ADAPTIVE_INSTR_DONE
