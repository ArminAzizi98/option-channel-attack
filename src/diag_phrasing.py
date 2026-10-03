"""How the policy is worded decides whether the gate fails open."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", f"{ROOT}/hf")
import numpy as np
from guardlab import guardbench as G, models as Mo, metrics as Me
from guardlab.config import ROOT

model = Mo.load(sys.argv[1] if len(sys.argv) > 1 else "laya-td")
print(f"{'policy':16s} {'phrasing':10s} {'acc':>7s} {'fail_open':>10s} {'fail_closed':>12s} {'block_rate':>11s}")
for pid in G.POLICY_IDS:
    items = G.build_items(pid, n=300, seed=0)
    states = [i.state for i in items]; gold = np.array([i.gold for i in items])
    for ph in ["positive", "explicit", "semantic"]:
        q, cn = G.question(pid, "aligned", phrasing=ph)
        r = model.score(states, q, cn)
        s = Me.rates(r["probs"], gold)
        print(f"{pid:16s} {ph:10s} {s['acc']:7.4f} {s['fail_open']:10.4f} "
              f"{s['fail_closed']:12.4f} {s['block_rate']:11.4f}")
    print()
