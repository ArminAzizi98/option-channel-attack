"""E5: adaptive attacks. Does each defense survive an attacker who knows about it?"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guardlab.config import ROOT
os.environ.setdefault("HF_HOME", f"{ROOT}/hf")
import numpy as np
from guardlab import guardbench as G, models as Mo, metrics as Me, attacks as A
from guardlab import defenses as D, adaptive as Ad


def asr_open(clean, adv, gold):
    cb = (gold == 1) & (clean.argmax(1) == 1)
    return (float((adv.argmax(1)[cb] == 0).mean()) if cb.sum() else float("nan")), int(cb.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya-td")
    ap.add_argument("--policies", default="unauth_irrev,secret,exfil,exfil_secret,escape,escape_or_exfil")
    ap.add_argument("--phrasing", default="positive")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=f"{ROOT}/runs/adaptive")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    done_path = os.path.join(a.out, f"{a.model}.json")
    if os.path.exists(done_path):
        print(f"skip {a.model}: {done_path} already written")
        return
    model = Mo.load(a.model)
    rows = []

    for pid in a.policies.split(","):
        items = G.build_items(pid, n=a.n, seed=0)
        gold = np.array([it.gold for it in items])
        fields = D.POLICY_FIELDS[pid]
        print(f"\n=== {pid} ({a.model}) ===", flush=True)

        # ---------------- D5 vs its own adaptive attack ----------------
        pc5, _ = D.defend_field_structured(model, items, pid, a.phrasing)
        base_fo = float((pc5.argmax(1)[gold == 1] == 0).mean())
        # non-adaptive: the old attack writes into the tool output, which D5 drops
        old_adv = A.attack_persuasion(items, 3)
        class _I:  # lightweight item view carrying the attacked text
            def __init__(self, st, uid): self.state, self.uid = st, uid
        old_items = [_I(s, it.uid) for s, it in zip(old_adv, items)]
        p_old, _ = D.defend_field_structured(model, old_items, pid, a.phrasing)
        fo_old, nb = asr_open(pc5, p_old, gold)
        # adaptive: write into the fields D5 keeps, leaving every attribute unchanged
        for k in (1, 2):
            adv = Ad.attack_field_injection(items, fields, k=k)
            adv_items = [_I(s, it.uid) for s, it in zip(adv, items)]
            p_new, _ = D.defend_field_structured(model, adv_items, pid, a.phrasing)
            fo_new, _ = asr_open(pc5, p_new, gold)
            print(f"  D5 clean fail_open={base_fo:.4f} | non-adaptive asr_open={fo_old:.4f} "
                  f"| ADAPTIVE field-inject k={k} asr_open={fo_new:.4f}  (n_blocked={nb})",
                  flush=True)
            rows.append({"policy": pid, "model": a.model, "defense": "D5 field-structured",
                         "clean_fail_open": base_fo, "asr_nonadaptive": fo_old,
                         "asr_adaptive": fo_new, "k": k, "n_blocked": nb})

        # ---------------- D4+D6 vs the dual-rendering attack ----------------
        q1, cn1 = D.question_with_labels(pid, "block", "allow", a.phrasing)
        q2, cn2 = D.question_with_labels(pid, "A", "B", a.phrasing)
        p1c = model.score([it.state for it in items], q1, cn1)["probs"]
        p2c = model.score([it.state for it in items], q2, cn2)["probs"]
        flag_c = p1c.argmax(1) != p2c.argmax(1)
        base = D.apply_fail_closed(p1c, flag_c)
        pool = A.FILLER + A.PERSUASION
        adv2, chosen, nq = Ad.attack_dual_rendering(items, model, q1, cn1, q2, cn2,
                                                    pool, budget=3)
        p1a = model.score(adv2, q1, cn1)["probs"]
        p2a = model.score(adv2, q2, cn2)["probs"]
        adv_gated = D.apply_fail_closed(p1a, p1a.argmax(1) != p2a.argmax(1))
        fo_dual, nb2 = asr_open(base, adv_gated, gold)
        # single-rendering greedy for comparison
        adv1, _, _ = A.attack_greedy(items, model, q1, cn1, pool=pool, budget=3)
        p1b = model.score(adv1, q1, cn1)["probs"]
        p2b = model.score(adv1, q2, cn2)["probs"]
        single_gated = D.apply_fail_closed(p1b, p1b.argmax(1) != p2b.argmax(1))
        fo_single, _ = asr_open(base, single_gated, gold)
        print(f"  D4+D6 single-rendering greedy asr_open={fo_single:.4f} | "
              f"ADAPTIVE dual-rendering asr_open={fo_dual:.4f}  "
              f"(n_blocked={nb2}, {nq} queries)", flush=True)
        rows.append({"policy": pid, "model": a.model, "defense": "D4+D6 disagreement",
                     "asr_nonadaptive": fo_single, "asr_adaptive": fo_dual,
                     "n_blocked": nb2, "queries": nq, "suffix": chosen})

    json.dump(rows, open(os.path.join(a.out, f"{a.model}.json"), "w"), indent=2)
    print(f"\nwrote {a.out}/{a.model}.json")


if __name__ == "__main__":
    main()
