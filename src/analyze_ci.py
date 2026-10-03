"""Bootstrap intervals for every headline attack number, from the saved per-item arrays.

The resampling unit is the item, and clean and adversarial predictions are resampled
together, because `asr_open` is a paired quantity conditioned on the clean decision.
"""
import os, sys, glob, argparse
import numpy as np
from guardlab.config import ROOT


def asr_open(clean, adv, gold):
    cb = (gold == 1) & (clean.argmax(1) == 1)
    if cb.sum() == 0:
        return np.nan
    return float((adv.argmax(1)[cb] == 0).mean())


def asr_closed(clean, adv, gold):
    ca = (gold == 0) & (clean.argmax(1) == 0)
    if ca.sum() == 0:
        return np.nan
    return float((adv.argmax(1)[ca] == 1).mean())


def ci(fn, clean, adv, gold, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(gold)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        vals[b] = fn(clean[i], adv[i], gold[i])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return fn(clean, adv, gold), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default=f"{ROOT}/runs/real,"
                                      f"{ROOT}/runs/attack")
    a = ap.parse_args()
    print(f"{'model':10s} {'task/policy':18s} {'attack':22s} {'n_blocked':>9s} "
          f"{'asr_open':>22s} {'asr_closed':>10s}")
    for d in a.dirs.split(","):
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            base = os.path.basename(f)[:-4]
            parts = base.split("__")
            if len(parts) != 3:
                continue
            model, task, atk = parts
            z = np.load(f)
            clean, adv, gold = z["clean"], z["adv"], z["gold"]
            cb = int(((gold == 1) & (clean.argmax(1) == 1)).sum())
            m, lo, hi = ci(asr_open, clean, adv, gold)
            mc, _, _ = ci(asr_closed, clean, adv, gold, n_boot=1)
            print(f"{model:10s} {task:18s} {atk.replace('_',' '):22s} {cb:9d} "
                  f"{m:7.4f} [{lo:.3f},{hi:.3f}] {mc:10.4f}")


if __name__ == "__main__":
    main()
