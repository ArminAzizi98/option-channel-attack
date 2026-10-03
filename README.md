# One Word Opens the Gate

Code and cached results for *One Word Opens the Gate: The Option-Channel Attack on Typed
Decision Models as Agent Guardrails*.

A **typed decision model** reads a piece of text and returns a probability over options that
the caller defines, each with a short written definition, and it generates no text. Recent work
places these models in agent systems as guardrails: the component that reads a proposed tool
call or an incoming message and decides whether to allow it. This repository evaluates seven
open-weight models in that role.

## What the paper finds

**The option channel.** Renaming the permissive option, while leaving its definition and the
text being judged untouched, drives the gate open. On Qwen2.5-7B, which blocks every violation
on four of six policies, naming that option `remediate` takes the fail-open rate to 100% on
**all six**. Ten labels tried at random reach 89%.

**It is a calling-library property, not a model property.** `von` and `rlcd` build the sequence
they encode from the option definitions alone, so the label never reaches the network. Across
19 cells and 198 candidate labels each, their fail-open rate is identical to four decimal
places whether the naming is honest or adversarial.

**Benign text opens gates too.** Six lines of routine server log text that say nothing about
the policy take one gate from 0% to 63% fail-open.

**Confidence does not help.** Over 1410 decisions that an attack reversed, the margin is 0.149
before the attack and 0.151 after, and *wider* after in 53% of them. Escalating low-confidence
cases cannot find a reversed decision, because a reversed decision is not low-confidence.

**Every defense is defeated**, either by an attacker who targets its mechanism or by
attacker-controlled text. Parsing each policy field into a typed value does remove one attack
completely, but the same parsing makes the model unnecessary: a deterministic rule over those
values is correct on 100% of items and resists both attacks.

## Reproducing the paper without a GPU

All cached run artifacts are committed under `runs/`, so every table, figure and number in the
paper regenerates in seconds:

```bash
pip install -r requirements.txt
PYTHONPATH=src python src/make_tables.py --out tables   # 14 LaTeX tables + 63 macros
PYTHONPATH=src python src/plots.py                      # 7 figures into figs/
PYTHONPATH=src python src/analyze_optsweep.py           # strata, query budget, transfer
```

No number in the paper is typed by hand. Every one is emitted as a LaTeX macro by
`src/make_tables.py` and read from the cached runs.

## Re-running the experiments

Needs one GPU and the model weights, which are fetched from the Hugging Face Hub on first use.

```bash
export JEVSEC_ROOT=/path/for/outputs        # optional; defaults to the repo
PYTHONPATH=src python src/run_clean.py    --model laya-td
PYTHONPATH=src python src/run_attacks.py  --model laya-td
PYTHONPATH=src python src/run_optsweep.py --model laya-td --n 200   # 198-label sweep
PYTHONPATH=src python src/run_real.py     --model laya-td --n 300   # public screening tasks
PYTHONPATH=src python src/run_defenses.py --model laya-td
PYTHONPATH=src python src/run_adaptive.py --model laya-td
```

Installing `laya`, `von` or `rlcd` pulls `nvidia-cudnn-cu13`, which breaks ModernBERT attention
against some torch builds. If you hit `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`, uninstall it
from the environment.

## Benchmarks

**GuardBench** (`src/guardlab/guardbench.py`) is generated, not shipped. Each item is a proposed
agent tool call built from six attributes, with the correct decision computed from those
attributes rather than annotated, so there is no label noise and no overlap with any model's
training data. Six policies span single-condition, conjunctive, disjunctive and negated rules,
and every trace carries an explicit attacker-controlled span.

The **public screening tasks** are third-party datasets and are *not* redistributed here. They
are downloaded from the Hugging Face Hub on first use: `deepset/prompt-injections`,
`jackhhao/jailbreak-classification` and `lmsys/toxic-chat`.

## Layout

```
src/guardlab/      benchmark, models, attacks, defenses, metrics
src/run_*.py       one experiment each; all write JSON and .npz into runs/
src/make_tables.py every LaTeX table and macro in the paper
src/plots.py       every figure
runs/              cached results (committed, so the paper rebuilds offline)
```

## Citation

```bibtex
@misc{azizi2026oneword,
  title  = {One Word Opens the Gate: The Option-Channel Attack on Typed Decision Models
            as Agent Guardrails},
  author = {Azizi, Seyedarmin and Pedram, Massoud},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## License

MIT, see `LICENSE`.
