#!/bin/bash
cd "$(dirname "$0")/.."          # repository root
# activate your environment first, e.g. source venv/bin/activate
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONPATH=src HF_HOME=hf
python src/run_optsweep.py --model qwen7b --n 200 --out runs/optsweep || echo "FAILED qwen7b"
echo OPTSWEEP_QWEN_DONE
