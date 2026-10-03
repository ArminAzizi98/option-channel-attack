"""Triage: use the gate as a pre-filter that escalates low-margin items instead of deciding.

The paper recommends this use but never measured it. The threshold is calibrated on clean
traffic, because that is all a deployer can see, and then evaluated under attack.
"""
import argparse, glob, json, os
import numpy as np
from guardlab.metrics import boot_ci
from guardlab.config import ROOT

BUDGETS = [0.0, 0.05, 0.10, 0.20, 0.50]


def margin(p):
    return np.abs(p[:, 1] - 0.5)


def residual(probs, gold, keep):
    """Error rates counting only items the gate decided itself (keep); escalated items are
    assumed to be resolved correctly by whatever they are escalated to."""
    pred = probs.argmax(1)
    blk, alw = gold == 1, gold == 0
    fo_num = int(((pred == 0) & blk & keep).sum())
    fc_num = int(((pred == 1) & alw & keep).sum())
    return (fo_num / int(blk.sum()) if blk.any() else float("nan"),
            fc_num / int(alw.sum()) if alw.any() else float("nan"))


def cell(f):
    model, task, attack = os.path.basename(f)[:-4].split("__")
    d = np.load(f, allow_pickle=True)
    clean, adv, gold = d["clean"], d["adv"], d["gold"]
    mc, ma = margin(clean), margin(adv)
    rows = []
    for b in BUDGETS:
        # threshold set so that exactly b of CLEAN traffic is escalated
        thr = -1.0 if b == 0 else float(np.quantile(mc, b))
        keep_c, keep_a = mc > thr, ma > thr
        fo_c, fc_c = residual(clean, gold, keep_c)
        fo_a, fc_a = residual(adv, gold, keep_a)
        rows.append({
            "budget": b, "threshold": thr,
            "esc_clean": float((~keep_c).mean()), "esc_attacked": float((~keep_a).mean()),
            "fo_clean": fo_c, "fc_clean": fc_c,
            "fo_attacked": fo_a, "fc_attacked": fc_a,
        })
    return {"model": model, "task": task, "attack": attack, "n": int(len(gold)), "rows": rows}


def flip_margins(files):
    """On items a successful attack reversed, the margin before and after it.

    If an attack left its victims looking uncertain, the margin after would be small and
    confidence-based escalation would find them.
    """
    per, tot = [], {"n": 0, "mb": 0.0, "ma": 0.0, "wider": 0}
    for f in files:
        model, task, attack = os.path.basename(f)[:-4].split("__")
        d = np.load(f, allow_pickle=True)
        c, a, g = d["clean"], d["adv"], d["gold"]
        flip = (g == 1) & (c.argmax(1) == 1) & (a.argmax(1) == 0)
        if flip.sum() < 5:
            continue
        mb, ma = margin(c[flip]), margin(a[flip])
        per.append({"model": model, "task": task, "attack": attack, "n": int(flip.sum()),
                    "margin_before": float(mb.mean()), "margin_after": float(ma.mean()),
                    "frac_wider": float((ma > mb).mean())})
        tot["n"] += int(flip.sum()); tot["mb"] += float(mb.sum())
        tot["ma"] += float(ma.sum()); tot["wider"] += int((ma > mb).sum())
    tot = {"n_flips": tot["n"], "margin_before": tot["mb"] / tot["n"],
           "margin_after": tot["ma"] / tot["n"], "frac_wider": tot["wider"] / tot["n"]}
    return {"per_cell": per, "pooled": tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=f"{ROOT}/runs/real")
    ap.add_argument("--out", default=f"{ROOT}/runs/triage")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cells = [cell(f) for f in sorted(glob.glob(os.path.join(a.runs, "*.npz")))]
    with open(os.path.join(a.out, "triage.json"), "w") as fh:
        json.dump(cells, fh, indent=1)
    fm = flip_margins(sorted(glob.glob(os.path.join(a.runs, "*.npz"))))
    with open(os.path.join(a.out, "margins.json"), "w") as fh:
        json.dump(fm, fh, indent=1)
    q = fm["pooled"]
    print(f"flipped items={q['n_flips']}  margin before={q['margin_before']:.3f} "
          f"after={q['margin_after']:.3f}  wider after in {q['frac_wider']:.1%} of flips\n")
    print(f"{len(cells)} cells -> {a.out}/triage.json\n")
    for b in BUDGETS:
        fo0, fob, esc = [], [], []
        for c in cells:
            r = [x for x in c["rows"] if x["budget"] == b][0]
            z = [x for x in c["rows"] if x["budget"] == 0.0][0]
            if np.isfinite(r["fo_attacked"]):
                fo0.append(z["fo_attacked"]); fob.append(r["fo_attacked"]); esc.append(r["esc_attacked"])
        m_fo, lo, hi = boot_ci(fob)
        print(f"clean budget {b:>5.0%} | attacked fail-open {np.mean(fo0):6.1%} -> "
              f"{m_fo:6.1%} [{lo:.1%},{hi:.1%}] | items escalated under attack {np.mean(esc):6.1%}")


if __name__ == "__main__":
    main()
