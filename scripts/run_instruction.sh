#!/bin/bash
cd "$(dirname "$0")/.."          # repository root
# activate your environment first, e.g. source venv/bin/activate
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONPATH=src HF_HOME=hf
for m in laya-td laya-en laya-ml qwen7b; do
  echo "######## $m $(date +%H:%M:%S)"
  python src/run_instruction.py --model "$m" --n 200 --out runs/instruction || echo "FAILED $m"
done
echo INSTRUCTION_ALL_DONE
