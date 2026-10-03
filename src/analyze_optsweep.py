"""Analyse the stratified label sweep: strata, query budget, cross-model transfer."""
import argparse, glob, json, os
import numpy as np
from guardlab.config import ROOT

HONEST = "allow"


def load(d, key):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        j = json.load(open(f))
        out[j["model"]] = j["rows"]
    return out


def per_stratum(rows, key):
    """Mean lift over the honest naming, by stratum."""
    units = sorted({r[key] for r in rows})
    hon = {u: [r["fail_open"] for r in rows if r[key] == u and r["allow_label"] == HONEST]
           for u in units}
    out = {}
    for st in sorted({r["stratum"] for r in rows}):
        lifts, tops = [], []
        for u in units:
            if not hon[u]:
                continue
            h = hon[u][0]
            v = [r["fail_open"] for r in rows if r[key] == u and r["stratum"] == st]
            if not v:
                continue
            lifts.append(float(np.mean(v)) - h)
            tops.append(max(v) - h)
        out[st] = {"mean_lift": float(np.mean(lifts)), "best_lift": float(np.mean(tops))}
    return out, {u: hon[u][0] for u in units if hon[u]}


def query_budget(rows, key, hon, ks=(1, 2, 5, 10, 20, 50, 100, 198), n_perm=400, seed=0):
    """Expected best fail-open after k labels tried in a random order."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in ks:
        per_unit = []
        for u, h in hon.items():
            v = np.array([r["fail_open"] for r in rows if r[key] == u])
            if len(v) < k:
                continue
            idx = np.argsort(rng.random((n_perm, len(v))), axis=1)[:, :k]
            per_unit.append(float(v[idx].max(axis=1).mean()))
        if per_unit:
            out[k] = float(np.mean(per_unit))
    return out


def transfer(models, key, topk=5):
    """Source model's best labels, evaluated on the target model."""
    lab = {}
    for m, rows in models.items():
        units = sorted({r[key] for r in rows})
        best = {}
        for u in units:
            v = sorted((r for r in rows if r[key] == u), key=lambda r: -r["fail_open"])
            best[u] = [r["allow_label"] for r in v[:topk]]
        lab[m] = best
    mat = {}
    for src in models:
        mat[src] = {}
        for tgt, rows in models.items():
            idx = {(r[key], r["allow_label"]): r["fail_open"] for r in rows}
            hon = {r[key]: r["fail_open"] for r in rows if r["allow_label"] == HONEST}
            vals = []
            for u, labs in lab[src].items():
                got = [idx[(u, l)] for l in labs if (u, l) in idx]
                if got and u in hon:
                    vals.append(max(got) - hon[u])
            mat[src][tgt] = float(np.mean(vals)) if vals else float("nan")
    return mat


def report(d, key, name):
    models = load(d, key)
    if not models:
        print(f"[{name}] no data yet"); return
    print(f"\n########## {name}  ({len(models)} models: {', '.join(models)})")
    for m, rows in models.items():
        st, hon = per_stratum(rows, key)
        print(f"\n-- {m}: honest fail-open per {key} = "
              f"{ {u: round(v,3) for u,v in hon.items()} }")
        print(f"   {'stratum':<20}{'mean lift':>11}{'best lift':>11}")
        for s, v in sorted(st.items(), key=lambda x: -x[1]["best_lift"]):
            print(f"   {s:<20}{v['mean_lift']:>+11.3f}{v['best_lift']:>+11.3f}")
        qb = query_budget(rows, key, hon)
        print("   query budget k -> expected best fail-open: " +
              "  ".join(f"{k}:{v:.3f}" for k, v in qb.items()))
    if len(models) > 1:
        print(f"\n-- transfer (rows = source of the labels, cols = model attacked), "
              f"mean lift over honest:")
        mat = transfer(models, key)
        ms = list(models)
        print("   " + " " * 12 + "".join(f"{m:>12}" for m in ms))
        for src in ms:
            print(f"   {src:<12}" + "".join(f"{mat[src][t]:>+12.3f}" for t in ms))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default=f"{ROOT}/runs/optsweep")
    ap.add_argument("--real", default=f"{ROOT}/runs/optsweep_real")
    a = ap.parse_args()
    report(a.synth, "policy", "GuardBench")
    report(a.real, "task", "public screening tasks")
