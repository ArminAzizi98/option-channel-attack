"""E1: clean guardrail behaviour by label condition, with errors split by direction."""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", f"{ROOT}/hf")
import numpy as np
from guardlab import guardbench as G, models as Mo, metrics as Me
from guardlab.config import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="laya-td,von")
    ap.add_argument("--conditions", default="aligned,neutral,unrelated,misleading")
    ap.add_argument("--phrasings", default="positive,explicit,semantic")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{ROOT}/runs/clean")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    for mname in a.models.split(","):
        print(f"\n[{time.strftime('%H:%M:%S')}] loading {mname}", flush=True)
        model = Mo.load(mname)
        for ph in a.phrasings.split(","):
          for cond in a.conditions.split(","):
            for pid in G.POLICY_IDS:
                p = os.path.join(a.out, f"{mname}__{pid}__{ph}__{cond}.npz")
                if os.path.exists(p):
                    continue
                items = G.build_items(pid, n=a.n, seed=a.seed)
                qspec, cn = G.question(pid, cond, phrasing=ph)
                states = [it.state for it in items]
                gold = np.array([it.gold for it in items])
                t0 = time.time()
                r = model.score(states, qspec, cn)
                np.savez_compressed(p, probs=r["probs"], gold=gold,
                                    confidence=r["confidence"],
                                    uids=np.array([it.uid for it in items]))
                st = Me.rates(r["probs"], gold)
                print(f"  {pid:18s} {ph:9s} {cond:11s} acc={st['acc']:.4f} "
                      f"fail_open={st['fail_open']:.4f} fail_closed={st['fail_closed']:.4f} "
                      f"block_rate={st['block_rate']:.4f}  {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
