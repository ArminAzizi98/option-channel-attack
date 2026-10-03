"""E2: attack a working gate and measure success in the unsafe direction only."""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", f"{ROOT}/hf")
import numpy as np
from guardlab import guardbench as G, models as Mo, metrics as Me, attacks as A
from guardlab.config import ROOT


def build_q(policy_id, block_label, allow_label, phrasing="positive"):
    table = G.PHRASINGS[phrasing]
    _, block_def, allow_def, _, _ = next(p for p in table if p[0] == policy_id)
    return ({"type": "choice", "instructions": G.INSTRUCTIONS,
             "criteria": {allow_label: allow_def, block_label: block_def}},
            [allow_label, block_label])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya-td")
    ap.add_argument("--policies", default="secret,unauth_irrev,exfil,exfil_secret")
    ap.add_argument("--phrasing", default="positive")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=f"{ROOT}/runs/attack")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = Mo.load(a.model)
    results = []

    for pid in a.policies.split(","):
        items = G.build_items(pid, n=a.n, seed=0)
        gold = np.array([it.gold for it in items])
        qspec, cn = G.question(pid, "aligned", phrasing=a.phrasing)
        clean = model.score([it.state for it in items], qspec, cn)["probs"]
        base = Me.rates(clean, gold)
        print(f"\n=== {pid}  ({a.model}, {a.phrasing}) ===", flush=True)
        print(f"  clean: acc={base['acc']:.4f} fail_open={base['fail_open']:.4f} "
              f"fail_closed={base['fail_closed']:.4f} correctly_blocked="
              f"{int(((gold==1)&(clean.argmax(1)==1)).sum())}", flush=True)

        for name, maker in [("distraction k=1", lambda: A.attack_distraction(items, 1)),
                            ("distraction k=3", lambda: A.attack_distraction(items, 3)),
                            ("distraction k=6", lambda: A.attack_distraction(items, 6)),
                            ("persuasion k=1", lambda: A.attack_persuasion(items, 1)),
                            ("persuasion k=3", lambda: A.attack_persuasion(items, 3))]:
            adv = maker()
            p = model.score(adv, qspec, cn)["probs"]
            st = Me.attack_rates(clean, p, gold)
            np.savez_compressed(
                os.path.join(a.out, f"{a.model}__{pid}__{name.replace(' ', '_')}.npz"),
                clean=clean, adv=p, gold=gold)
            print(f"  {name:16s} asr_open={st['asr_open']:.4f} asr_closed={st['asr_closed']:.4f} "
                  f"dP_block={st['dP_block']:+.4f}", flush=True)
            results.append({"policy": pid, "attack": name, "model": a.model, **st})

        # option-channel: no trace text is touched at all
        oc = A.attack_option_channel(items, model, pid,
                                     lambda p, b, al: build_q(p, b, al, a.phrasing))
        rows = sorted(oc["rows"], key=lambda r: -r["fail_open"])
        honest = [r for r in oc["rows"] if r["block_label"] == "block"
                  and r["allow_label"] == "allow"][0]
        print(f"  option-channel   honest fail_open={honest['fail_open']:.4f} -> "
              f"worst {rows[0]['fail_open']:.4f} with labels "
              f"(block->'{rows[0]['block_label']}', allow->'{rows[0]['allow_label']}')", flush=True)
        results.append({"policy": pid, "attack": "option_channel", "model": a.model,
                        "honest_fail_open": honest["fail_open"],
                        "worst_fail_open": rows[0]["fail_open"],
                        "worst_labels": [rows[0]["block_label"], rows[0]["allow_label"]],
                        "all_rows": oc["rows"]})

    with open(os.path.join(a.out, f"{a.model}__{a.phrasing}.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {a.out}/{a.model}__{a.phrasing}.json")


if __name__ == "__main__":
    main()
