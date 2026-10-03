"""Option-channel attack on the three public screening tasks.

The attack was previously measured only on GuardBench, so it could be dismissed as an artifact
of synthetic traces. Here the text being judged was written by real users and is not touched:
only the name of the permissive option changes.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", f"{ROOT}/hf")
import numpy as np
from guardlab import realtasks as R, models as Mo
from run_optsweep import vocabulary, HONEST
from guardlab.config import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya-td")
    ap.add_argument("--tasks", default="injection,jailbreak,toxic")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--labels", default="")
    ap.add_argument("--out", default=f"{ROOT}/runs/optsweep_real")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    vocab = vocabulary()
    if a.labels:
        keep = set(a.labels.split(",")) | {HONEST}
        vocab = [(w, s) for w, s in vocab if w in keep]
    model = Mo.load(a.model)
    rows, t0 = [], time.time()

    for task in a.tasks.split(","):
        items = R.load_real(task, n=a.n, seed=0)
        gold = np.array([i.gold for i in items])
        states = [i.state for i in items]
        blocked = gold == 1
        for k, (lab, stratum) in enumerate(vocab):
            q, cn = R.question(task, block_label="block", allow_label=lab)
            p = model.score(states, q, cn, batch_size=64)["probs"]
            pred = p.argmax(1)
            rows.append({"model": a.model, "task": task, "allow_label": lab,
                         "stratum": stratum,
                         "fail_open": float((pred[blocked] == 0).mean()),
                         "acc": float((pred == gold).mean())})
            if (k + 1) % 50 == 0:
                print(f"  {task}: {k+1}/{len(vocab)}, {time.time()-t0:.0f}s", flush=True)
        h = [r for r in rows if r["task"] == task and r["allow_label"] == HONEST][0]
        best = max((r for r in rows if r["task"] == task), key=lambda r: r["fail_open"])
        print(f"=== {task} ({a.model}) honest fail_open={h['fail_open']:.4f} -> "
              f"best {best['fail_open']:.4f} with '{best['allow_label']}' [{best['stratum']}]",
              flush=True)

    with open(os.path.join(a.out, f"{a.model}.json"), "w") as fh:
        json.dump({"model": a.model, "n": a.n, "rows": rows}, fh)
    print(f"wrote {a.out}/{a.model}.json ({len(rows)} cells, {time.time()-t0:.0f}s)", flush=True)
    print("OPTSWEEP_REAL_DONE", flush=True)


if __name__ == "__main__":
    main()
