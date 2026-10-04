#!/bin/bash
cd "$(dirname "$0")/.."          # repository root
export HF_HOME=hf
export HF_DATASETS_CACHE=data
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONHASHSEED=0
POL=unauth_irrev,secret,exfil,exfil_secret,escape,escape_or_exfil
echo "### STAGE 1: attacks (feeds tab_attacks, tab_option, fig2, fig3)"
for M in laya-td von rlcd laya-en laya-ml qwen7b; do
  ./venv/bin/python src/run_attacks.py --model $M --policies $POL --n 300 --out runs/attack
done
echo "### STAGE 2: defenses + matched-baseline control"
for M in laya-td von laya-en laya-ml rlcd; do
  ./venv/bin/python src/run_defenses.py --model $M --policies $POL --n 300 --out runs/defense
done
for M in laya-td von laya-en; do
  for P in unauth_irrev secret exfil exfil_secret escape escape_or_exfil; do
    ./venv/bin/python src/run_tradeoff.py --model $M --policy $P --n 300 --out runs/tradeoff
  done
done
echo "### STAGE 3: adaptive attacks"
for M in laya-td von rlcd laya-en laya-ml; do
  ./venv/bin/python src/run_adaptive.py --model $M --n 300 --out runs/adaptive
done
echo "### STAGE 4: clean phrasing sweep"
for M in laya-td von rlcd; do
  ./venv/bin/python src/run_clean.py --models $M --n 300 --out runs/clean
done
echo REGEN_DONE
