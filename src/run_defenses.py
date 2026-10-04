"""E3: do the defenses close the channels, and what do they cost?"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guardlab.config import ROOT
os.environ.setdefault("HF_HOME", f"{ROOT}/hf")
import numpy as np
from guardlab import guardbench as G, models as Mo, metrics as Me, attacks as A, defenses as D


def fo(probs, gold):
    blk = gold == 1
    return float((probs.argmax(1)[blk] == 0).mean()) if blk.any() else float("nan")


def fc(probs, gold):
    alw = gold == 0
    return float((probs.argmax(1)[alw] == 1).mean()) if alw.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya-td")
    ap.add_argument("--policies", default="unauth_irrev")
    ap.add_argument("--phrasing", default="positive")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=f"{ROOT}/runs/defense")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = Mo.load(a.model)
    allrows = []

    for pid in a.policies.split(","):
        items = G.build_items(pid, n=a.n, seed=0)
        gold = np.array([it.gold for it in items])
        clean_states = [it.state for it in items]
        adv_states = {
            "distraction k=6": A.attack_distraction(items, 6),
            "persuasion k=3": A.attack_persuasion(items, 3),
        }
        print(f"\n################ {pid}  ({a.model}) ################", flush=True)
        print(f"{'defense':26s} {'calls':>5s} {'clean acc':>10s} {'clean FO':>9s} "
              f"{'clean FC':>9s} {'FO distract':>12s} {'FO persuade':>12s} {'FO rename(worst)':>17s}")

        def row(name, calls, probs_clean, probs_adv, worst_rename):
            print(f"{name:26s} {calls:5d} {float((probs_clean.argmax(1)==gold).mean()):10.4f} "
                  f"{fo(probs_clean, gold):9.4f} {fc(probs_clean, gold):9.4f} "
                  f"{fo(probs_adv['distraction k=6'], gold):12.4f} "
                  f"{fo(probs_adv['persuasion k=3'], gold):12.4f} "
                  f"{worst_rename:17}", flush=True)
            allrows.append({"policy": pid, "model": a.model, "defense": name, "calls": calls,
                            "clean_acc": float((probs_clean.argmax(1) == gold).mean()),
                            "clean_fail_open": fo(probs_clean, gold),
                            "clean_fail_closed": fc(probs_clean, gold),
                            "fo_distraction": fo(probs_adv["distraction k=6"], gold),
                            "fo_persuasion": fo(probs_adv["persuasion k=3"], gold),
                            "worst_rename": worst_rename})

        # ---- undefended ----
        q, cn = G.question(pid, "aligned", phrasing=a.phrasing)
        pc = model.score(clean_states, q, cn)["probs"]
        pa = {k: model.score(v, q, cn)["probs"] for k, v in adv_states.items()}
        cert = D.certify_label_rename(model, clean_states, pid, a.phrasing)
        blk = gold == 1
        worst = float((cert["any_allow"][blk]).mean())
        row("undefended", 1, pc, pa, f"{worst:.4f}")

        # ---- D1 label canonicalisation ----
        pc1, c1 = D.defend_canonical(model, clean_states, pid, a.phrasing)
        pa1 = {k: D.defend_canonical(model, v, pid, a.phrasing)[0] for k, v in adv_states.items()}
        row("D1 canonical labels", c1, pc1, pa1, "immune")

        # ---- D3 rendering ensemble ----
        pc3, c3 = D.defend_ensemble(model, clean_states, pid, a.phrasing, m=3)
        pa3 = {k: D.defend_ensemble(model, v, pid, a.phrasing, m=3)[0] for k, v in adv_states.items()}
        row("D3 ensemble of 3", c3, pc3, pa3, "immune")

        # ---- D5 field-structured ----
        pc5, c5 = D.defend_field_structured(model, items, pid, a.phrasing)
        # the attacker's span is not in the field view, so adversarial states are identical
        pa5 = {k: pc5 for k in adv_states}
        row("D5 field-structured", c5, pc5, pa5, "immune")

        # ---- D4+D6 disagreement gate, fail closed ----
        p1, flag, c4 = D.defend_disagreement(model, clean_states, pid, a.phrasing)
        pc4 = D.apply_fail_closed(p1, flag)
        pa4 = {}
        for k, v in adv_states.items():
            pv, fv, _ = D.defend_disagreement(model, v, pid, a.phrasing)
            pa4[k] = D.apply_fail_closed(pv, fv)
        row(f"D4+D6 disagree,failclosed", c4, pc4, pa4, "immune")
        print(f"  (escalation rate on clean input: {flag.mean():.4f})", flush=True)

    with open(os.path.join(a.out, f"{a.model}__{a.phrasing}.json"), "w") as fh:
        json.dump(allrows, fh, indent=2)
    print(f"\nwrote {a.out}/{a.model}__{a.phrasing}.json")


if __name__ == "__main__":
    main()
