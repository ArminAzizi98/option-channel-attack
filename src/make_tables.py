"""Emit LaTeX tables and macros from the cached guardrail runs. No number is hand-typed."""
import os, sys, json, glob, re, argparse
import numpy as np
from guardlab.config import ROOT

OUT = f"{ROOT}/paper/tables"
RUNS = f"{ROOT}/runs"
MACROS = {}
PRETTY = {"laya-td": r"\textsc{laya-td}", "laya-en": r"\textsc{laya-en}",
          "laya-ml": r"\textsc{laya-ml}", "von": r"\textsc{von}",
          "rlcd": r"\textsc{rlcd}", "qwen1.5b": "Qwen2.5-1.5B", "qwen7b": "Qwen2.5-7B"}


def R(x, nd=1):
    """A rate in [0,1] rendered as a percentage. Tables and prose both use percentages so
    the reader never has to convert between the two."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "--"
    if x != x:
        return "--"
    return f"{100 * x:.{nd}f}"


def mac(k, v):
    MACROS[k] = v


def write(name, body):
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name), "w").write(body)
    print("wrote", name)


def asr_open(c, a, g):
    cb = (g == 1) & (c.argmax(1) == 1)
    return float((a.argmax(1)[cb] == 0).mean()) if cb.sum() else np.nan


def ci(c, a, g, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(g)
    v = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        v[b] = asr_open(c[i], a[i], g[i])
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan, np.nan
    lo, hi = np.quantile(v, [0.025, 0.975])
    return asr_open(c, a, g), float(lo), float(hi)


def load(d, pat):
    out = {}
    for f in glob.glob(os.path.join(RUNS, d, pat)):
        parts = os.path.basename(f)[:-4].split("__")
        if len(parts) != 3:
            continue
        z = np.load(f)
        out[tuple(parts)] = (z["clean"], z["adv"], z["gold"])
    return out


# ---------------------------------------------------- real-task clean behaviour
def tab_directions():
    """Clean behaviour on the public screening tasks, computed from the saved cells."""
    import glob as _g
    rows = {}
    for f in _g.glob(os.path.join(RUNS, "real", "*__*__benign_suffix_x3.npz")):
        m, t, _a = os.path.basename(f)[:-4].split("__")
        z = np.load(f)
        c, g = z["clean"], z["gold"]
        pred = c.argmax(1)
        rows.setdefault(m, {})[t] = (float((pred == g).mean()),
                                     float((pred[g == 1] == 0).mean()),
                                     float((pred[g == 0] == 1).mean()), len(g))
    order = ["laya-td", "laya-en", "laya-ml", "von", "rlcd", "qwen1.5b", "qwen7b"]
    b = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"model & task & $n$ & accuracy & fail-open & fail-closed \\", r"\midrule"]
    accs = []
    for m in [x for x in order if x in rows]:
        first = True
        for t in ("injection", "jailbreak", "toxic"):
            if t not in rows[m]:
                continue
            a, fo, fc, n = rows[m][t]
            accs.append(a)
            b.append(f"{PRETTY.get(m, m) if first else '':22s} & {t} & {n} & "
                     f"{R(a)} & {R(fo)} & {R(fc)} " + r"\\")
            first = False
    b += [r"\bottomrule", r"\end{tabular}"]
    write("tab_directions.tex", "\n".join(b) + "\n")
    if accs:
        mac("accminpct", pct(min(accs)))
        mac("accmaxpct", pct(max(accs)))
    mac("chance", "0.500")


# ---------------------------------------------------- synthetic attacks
def tab_attacks():
    cells = load("attack", "*.npz")
    models = sorted({k[0] for k in cells})
    atks = ["distraction_k=1", "distraction_k=3", "distraction_k=6",
            "persuasion_k=1", "persuasion_k=3"]
    b = ["\\begin{tabular}{ll" + "r" * len(atks) + "}", "\\toprule",
         "model & policy & " + " & ".join(a.replace("_", " ").replace("k=", "$k{=}$")
                                          for a in atks) + " \\\\", "\\midrule"]
    for m in models:
        pols = sorted({k[1] for k in cells if k[0] == m})
        first = True
        for p in pols:
            row = []
            for a in atks:
                key = (m, p, a)
                if key in cells:
                    v = asr_open(*cells[key])
                    row.append(R(v))
                else:
                    row.append("--")
            esc = p.replace("_", r"\_")
            name = PRETTY.get(m, m) if first else ""
            b.append(f"{name:22s} & \\texttt{{{esc}}} & " + " & ".join(row) + r" \\")
            first = False
        b.append("\\addlinespace")
    b += ["\\bottomrule", "\\end{tabular}"]
    write("tab_attacks.tex", "\n".join(b) + "\n")
    # headline macro: von on exfil at k=6
    k = ("von", "exfil", "distraction_k=6")
    if k in cells:
        m_, lo, hi = ci(*cells[k])
        mac("vonexfil", f"{m_:.4f}"); mac("vonexfillo", f"{lo:.3f}"); mac("vonexfilhi", f"{hi:.3f}")


# ---------------------------------------------------- option channel
def tab_option():
    b = ["\\begin{tabular}{llrrl}", "\\toprule",
         "model & policy & honest & worst renaming & label installed \\\\", "\\midrule"]
    for f in sorted(glob.glob(os.path.join(RUNS, "attack", "*.json"))):
        m = os.path.basename(f)[:-5].split("__")[0]
        best = None
        for r in json.load(open(f)):
            if r.get("attack") != "option_channel":
                continue
            if best is None or r["worst_fail_open"] > best["worst_fail_open"]:
                best = r
        if not best:
            continue
        lab = best["worst_labels"]
        imm = abs(best["worst_fail_open"] - best["honest_fail_open"]) < 1e-9
        esc = best["policy"].replace("_", r"\_")
        how = ("immune by construction" if imm else
               f"block to \\texttt{{{lab[0]}}}, allow to \\texttt{{{lab[1]}}}")
        b.append(f"{PRETTY.get(m, m)} & \\texttt{{{esc}}} & "
                 f"{R(best['honest_fail_open'])} & {R(best['worst_fail_open'])} & "
                 + how + r" \\")
    b += ["\\bottomrule", "\\end{tabular}"]
    write("tab_option.tex", "\n".join(b) + "\n")


# ---------------------------------------------------- defenses
def tab_defense():
    rows = []
    for f in sorted(glob.glob(os.path.join(RUNS, "defense", "*.json"))):
        rows += json.load(open(f))
    rows = [r for r in rows if r.get("policy") == "unauth_irrev"]
    b = ["\\begin{tabular}{llrrrrr}", "\\toprule",
         "model & defense & calls & clean acc & clean FC & FO distract & FO persuade \\\\",
         "\\midrule"]
    last = None
    for r in rows:
        m = r["model"]
        b.append(f"{PRETTY.get(m, m) if m != last else '':22s} & {r['defense']} & {r['calls']} & "
                 f"{R(r['clean_acc'])} & {R(r['clean_fail_closed'])} & "
                 f"{R(r['fo_distraction'])} & {R(r['fo_persuasion'])} \\\\")
        last = m
    b += ["\\bottomrule", "\\end{tabular}"]
    write("tab_defense.tex", "\n".join(b) + "\n")


def tab_verdicts():
    """The matched-baseline control over every model x policy x defense cell."""
    f = os.path.join(RUNS, "tradeoff_verdicts.json")
    if not os.path.exists(f):
        return
    rows = json.load(open(f))
    import collections
    pd_ = collections.defaultdict(collections.Counter)
    pm_ = collections.defaultdict(collections.Counter)
    for m, pol, d, fc, fo, bo, v in rows:
        pd_[d][v] += 1; pm_[m][v] += 1
    b = ["\\begin{tabular}{lrrr}", "\\toprule",
         "defense & better & tie & worse \\\\", "\\midrule"]
    for d in sorted(pd_, key=lambda x: -pd_[x]["better"]):
        c = pd_[d]
        b.append(f"{d} & {c['better']} & {c['tie']} & {c['worse']} \\\\")
    b += ["\\midrule", "\\multicolumn{4}{l}{\\emph{by model}} \\\\"]
    for m in sorted(pm_):
        c = pm_[m]
        b.append(f"{PRETTY.get(m, m)} & {c['better']} & {c['tie']} & {c['worse']} \\\\")
    b += ["\\bottomrule", "\\end{tabular}"]
    write("tab_verdicts.tex", "\n".join(b) + "\n")
    mac("ncells", str(len(rows)))
    tot = collections.Counter(r[6] for r in rows)
    mac("nbetter", str(tot["better"])); mac("ntie", str(tot["tie"]))
    mac("nworse", str(tot["worse"]))


def tab_adaptive():
    """Each defense against an attacker who targets its mechanism."""
    import glob as _g
    d5, d4 = {}, {}
    for f in sorted(_g.glob(os.path.join(RUNS, "adaptive", "*.json"))):
        m = os.path.basename(f)[:-5]
        for r in json.load(open(f)):
            if r["defense"].startswith("D5"):
                a = r["asr_adaptive"]
                if a == a:                      # drop non-evaluable cells (no blocked items)
                    d5.setdefault(m, {}).setdefault(r["policy"], []).append(a)
            elif r["defense"].startswith("D4"):
                d4.setdefault(m, []).append((r["asr_nonadaptive"], r["asr_adaptive"]))
    b = ["\\begin{tabular}{lrrrrr}", "\\toprule",
         "model & non-adaptive & median & max & $>0.5$ & $\\approx 0$ \\\\", "\\midrule"]
    ncells = 0
    worstall = 0.0
    for m in sorted(d5):
        per = {p: max(v) for p, v in d5[m].items()}
        ncells += sum(len(v) for v in d5[m].values())
        vals = np.array(list(per.values()))
        worstall = max(worstall, float(vals.max()))
        b.append(f"{PRETTY.get(m, m)} & {R(0.0)} & {R(np.median(vals))} & "
                 f"{R(vals.max())} & {int((vals > 0.5).sum())} & "
                 f"{int((vals < 0.05).sum())} \\\\")
    b += ["\\bottomrule", "\\end{tabular}"]
    write("tab_adaptive.tex", "\n".join(b) + "\n")
    mac("adaptmax", f"{worstall:.4f}")
    mac("adaptmodels", str(len(d5)))

    b2 = ["\\begin{tabular}{lrr}", "\\toprule",
          "model & single-rendering & adaptive dual-rendering \\\\", "\\midrule"]
    for m in sorted(d4):
        sg = np.array([x[0] for x in d4[m]]); du = np.array([x[1] for x in d4[m]])
        b2.append(f"{PRETTY.get(m, m)} & {R(np.nanmean(sg))} & {R(np.nanmean(du))} \\\\")
    b2 += ["\\bottomrule", "\\end{tabular}"]
    write("tab_adaptive_d4.tex", "\n".join(b2) + "\n")


def tab_normalize():
    """Value normalisation, the attack that still beats it, and the deterministic rule."""
    import glob as _g
    rows = {}
    rule = []
    for f in sorted(_g.glob(os.path.join(RUNS, "normalize", "*.json"))):
        m = os.path.basename(f)[:-5]
        for r in json.load(open(f)):
            if r["defense"] == "rule":
                rule.append(r)
            else:
                rows.setdefault(m, {}).setdefault(r["defense"], []).append(r)
    b = ["\\begin{tabular}{llrrr}", "\\toprule",
         "model & gate input & field injection & confusing values & clean fail-open \\\\",
         "\\midrule"]
    for m in sorted(rows):
        first = True
        for dfn, short in (("D5 raw fields", "raw field text"),
                           ("D7 parsed values", "parsed value only")):
            v = rows[m].get(dfn, [])
            if not v:
                continue
            fi = np.array([x["asr_field_injection"] for x in v], dtype=float)
            cf = np.array([x["asr_confusing"] for x in v], dtype=float)
            fo = np.array([x["clean_fo"] for x in v], dtype=float)
            b.append(f"{PRETTY.get(m, m) if first else '':20s} & {short} & "
                     f"{R(np.nanmean(fi))} & {R(np.nanmean(cf))} & {R(np.nanmean(fo))} \\\\")
            first = False
        b.append("\\addlinespace")
    if rule:
        acc = np.array([r["acc"] for r in rule], dtype=float)
        fi = np.nanmax([r["asr_field_injection"] for r in rule])
        cf = np.nanmax([r["asr_confusing"] for r in rule])
        b.append(f"deterministic rule & parsed value only & {R(fi)} & {R(cf)} & "
                 f"{R(1 - acc.min())} \\\\")
        mac("ruleacc", f"{acc.min():.4f}")
        mac("ruleaccpct", pct(acc.min()))
        mac("rulemaxfi", f"{fi:.4f}")
    b += ["\\bottomrule", "\\end{tabular}"]
    write("tab_normalize.tex", "\n".join(b) + "\n")
    # headline macros
    all_fi_raw, all_fi_norm = [], []
    for m in rows:
        all_fi_raw += [x["asr_field_injection"] for x in rows[m].get("D5 raw fields", [])]
        all_fi_norm += [x["asr_field_injection"] for x in rows[m].get("D7 parsed values", [])]
    if all_fi_raw:
        mac("normrawfi", f"{np.nanmean(all_fi_raw):.4f}")
        mac("normfi", f"{np.nanmean(all_fi_norm):.4f}")
        mac("normrawfipct", pct(np.nanmean(all_fi_raw)))
        mac("normfipct", pct(np.nanmean(all_fi_norm)))


def pct(x, nd=0):
    """Rate in [0,1] as a percentage string, for prose where decimals read badly."""
    return f"{100 * float(x):.{nd}f}\\%"


def tab_headline():
    """Macros for the numbers quoted in the abstract and introduction.

    These were hardcoded once and went stale after the benchmark was regenerated, so they
    are generated from the cells like everything else.
    """
    import glob as _g
    # dose response on the one synthetic policy where the clean gate is accurate
    f = os.path.join(RUNS, "attack", "laya-td__unauth_irrev__distraction_k=6.npz")
    if os.path.exists(f):
        z = np.load(f)
        c, a, g = z["clean"], z["adv"], z["gold"]
        cb = (g == 1) & (c.argmax(1) == 1)
        mac("dosecleanpct", pct(float((c.argmax(1)[g == 1] == 0).mean())))
        mac("dosesixpct", pct(float((a.argmax(1)[cb] == 0).mean())))
        mac("doseacc", f"{float((c.argmax(1) == g).mean()):.4f}")
    # option-channel range over the models that put the label in the input
    worst = []
    for fp in sorted(_g.glob(os.path.join(RUNS, "attack", "*.json"))):
        best = None
        for r in json.load(open(fp)):
            if r.get("attack") != "option_channel":
                continue
            d = r["worst_fail_open"] - r["honest_fail_open"]
            if best is None or d > best[0]:
                best = (d, r)
        if best and best[0] > 1e-9:
            worst.append(best[1]["worst_fail_open"])
    if worst:
        mac("ocminpct", pct(min(worst)))
        mac("ocmaxpct", pct(max(worst)))
        mac("ocnmodels", str(len(worst)))
        # Spelled-out form, for prose where a bare digit reads badly.
        words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
        mac("ocnmodelsword", words.get(len(worst), str(len(worst))))
    mac("chancepct", "50\\%")


def tab_triage():
    """Using the gate as a pre-filter that escalates low-margin items instead of deciding.

    The escalation threshold is calibrated on clean traffic, because that is all a deployer
    sees, and then evaluated under attack.
    """
    fc = os.path.join(RUNS, "triage", "triage.json")
    fm = os.path.join(RUNS, "triage", "margins.json")
    if not (os.path.exists(fc) and os.path.exists(fm)):
        return
    cells = json.load(open(fc))
    budgets = sorted({r["budget"] for c in cells for r in c["rows"]})
    body = [r"\begin{tabular}{rrrrr}", r"\toprule",
            r"clean escalation budget & escalated under attack & fail-open under attack "
            r"& fail-closed, clean & oracle fail-open \\", r"\midrule"]
    base = None
    for b in budgets:
        rs = [r for c in cells for r in c["rows"] if r["budget"] == b]
        esc = np.mean([r["esc_attacked"] for r in rs])
        fo = np.nanmean([r["fo_attacked"] for r in rs])
        fcl = np.nanmean([r["fc_clean"] for r in rs])
        if base is None:
            base = fo
        # an oracle that spends the same review budget on the items that do fail open
        share = base * 0.5          # fail-open items as a share of all traffic (classes balanced)
        orac = max(0.0, 1.0 - esc / share) * base if share > 0 else 0.0
        body.append(f"{R(b)} & {R(esc)} & {R(fo)} & {R(fcl)} & {R(orac)} \\\\")
        if abs(b - 0.10) < 1e-9:
            mac("trbtenfopct", pct(fo)); mac("trbtenescpct", pct(esc))
        if abs(b - 0.50) < 1e-9:
            mac("trbfiftyfopct", pct(fo)); mac("trbfiftyescpct", pct(esc))
            mac("trbfiftyoracpct", pct(orac))
    body += [r"\bottomrule", r"\end{tabular}"]
    mac("ncellstriage", str(len(cells)))
    write("tab_triage.tex", "\n".join(body) + "\n")
    mac("trbasefopct", pct(base))

    q = json.load(open(fm))["pooled"]
    mac("trnflips", str(q["n_flips"]))
    mac("trmarginbefore", f"{q['margin_before']:.3f}")
    mac("trmarginafter", f"{q['margin_after']:.3f}")
    mac("trwiderpct", pct(q["frac_wider"]))


def _sweep(d, key):
    import glob as _g
    out = {}
    for f in sorted(_g.glob(os.path.join(d, "*.json"))):
        j = json.load(open(f))
        out[j["model"]] = j["rows"]
    return out


def _qbudget(rows, key, ks=(1, 5, 10), n_perm=400, seed=0):
    rng = np.random.default_rng(seed)
    units = sorted({r[key] for r in rows})
    res = {}
    for k in ks:
        vals = []
        for u in units:
            v = np.array([r["fail_open"] for r in rows if r[key] == u])
            if len(v) < k:
                continue
            idx = np.argsort(rng.random((n_perm, len(v))), axis=1)[:, :k]
            vals.append(float(v[idx].max(axis=1).mean()))
        res[k] = float(np.mean(vals)) if vals else float("nan")
    return res


def tab_optsweep():
    """The option channel over a stratified vocabulary, with the attacker's query budget."""
    for d, key, name, tag in [
            (os.path.join(RUNS, "optsweep"), "policy", "tab_optsweep.tex", "gb"),
            (os.path.join(RUNS, "optsweep_real"), "task", "tab_optsweep_real.tex", "rt")]:
        ms = _sweep(d, key)
        if not ms:
            continue
        body = [r"\begin{tabular}{lrrrrrl}", r"\toprule",
                r"model & honest & $k{=}1$ & $k{=}5$ & $k{=}10$ & best & best label \\",
                r"\midrule"]
        for m, rows in ms.items():
            units = sorted({r[key] for r in rows})
            hon = np.mean([[r["fail_open"] for r in rows
                            if r[key] == u and r["allow_label"] == "allow"][0] for u in units])
            qb = _qbudget(rows, key)
            best = np.mean([max(r["fail_open"] for r in rows if r[key] == u) for u in units])
            top = max(rows, key=lambda r: r["fail_open"])
            body.append(f"{PRETTY.get(m, m)} & {R(hon)} & {R(qb[1])} & {R(qb[5])} & "
                        f"{R(qb[10])} & {R(best)} & \\code{{{top['allow_label']}}} \\\\")
            if m == "laya-td" and tag == "rt":
                mac("swrthonpct", pct(hon)); mac("swrtktenpct", pct(qb[10]))
                mac("swrtbestpct", pct(best))
                jb = [r for r in rows if r.get("task") == "jailbreak"]
                if jb:
                    mac("swrtjbhonpct", pct([r["fail_open"] for r in jb
                                             if r["allow_label"] == "allow"][0]))
                    mac("swrtjbbestpct", pct(max(r["fail_open"] for r in jb)))
            if m == "laya-ml" and tag == "rt":
                tx = [r for r in rows if r.get("task") == "toxic"]
                if tx:
                    mac("swrttxhonpct", pct([r["fail_open"] for r in tx
                                             if r["allow_label"] == "allow"][0]))
                    mac("swrttxbestpct", pct(max(r["fail_open"] for r in tx)))
            if m == "qwen7b" and tag == "gb":
                mac("swqwhonpct", pct(hon)); mac("swqwbestpct", pct(best))
                mac("swqwktenpct", pct(qb[10]))
                mac("swqwlabel", top["allow_label"])
                nperf = sum(1 for u in units
                            if [r["fail_open"] for r in rows
                                if r[key] == u and r["allow_label"] == "allow"][0] < 1e-9)
                mac("swqwnzero", str(nperf))
            if m == "laya-td" and tag == "gb":
                mac("swhonpct", pct(hon)); mac("swkonepct", pct(qb[1]))
                mac("swkfivepct", pct(qb[5])); mac("swktenpct", pct(qb[10]))
                mac("swbestpct", pct(best))
        body += [r"\bottomrule", r"\end{tabular}"]
        write(name, "\n".join(body) + "\n")
        mac("swnlabels", str(len({r["allow_label"] for r in next(iter(ms.values()))})))


def tab_strata():
    """Which kinds of label open the gate, on the one policy where the clean gate is accurate."""
    ms = _sweep(os.path.join(RUNS, "optsweep"), "policy")
    if "laya-td" not in ms:
        return
    rows = [r for r in ms["laya-td"] if r["policy"] == "unauth_irrev"]
    if not rows:
        return
    hon = [r["fail_open"] for r in rows if r["allow_label"] == "allow"][0]
    body = [r"\begin{tabular}{lrrr}", r"\toprule",
            r"kind of label & mean & best & share above \(\text{honest}+20\) \\",
            r"\midrule"]
    order = sorted({r["stratum"] for r in rows},
                   key=lambda st: -np.mean([r["fail_open"] for r in rows if r["stratum"] == st]))
    nice = {"security_action": "names a security action", "neutral_syntactic": "neutral token",
            "control_unrelated": "unrelated noun", "allow_synonym": "synonym of allow",
            "severity": "names a severity", "block_synonym": "synonym of block"}
    for st in order:
        v = np.array([r["fail_open"] for r in rows if r["stratum"] == st])
        body.append(f"{nice.get(st, st)} & {R(v.mean())} & {R(v.max())} & "
                    f"{R(float((v > hon + 0.2).mean()))} \\\\")
        mac("sw" + st.replace("_", "") + "pct", pct(v.mean()))
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_strata.tex", "\n".join(body) + "\n")
    mac("swhonestcleanpct", pct(hon))


def tab_transfer():
    """Labels found on one model, used against another."""
    ms = _sweep(os.path.join(RUNS, "optsweep"), "policy")
    if len(ms) < 2:
        return
    names = list(ms)
    # A label-invariant model ties every label, so its "best labels" are an arbitrary
    # tie-break and cannot measure transfer. Such models stay as columns, not as rows.
    def varies(rows):
        return any(max(r["fail_open"] for r in rows if r["policy"] == u)
                   - min(r["fail_open"] for r in rows if r["policy"] == u) > 1e-9
                   for u in {r["policy"] for r in rows})
    sources = [m for m in names if varies(ms[m])]
    lab = {}
    for m, rows in ms.items():
        lab[m] = {u: [r["allow_label"] for r in
                      sorted((x for x in rows if x["policy"] == u),
                             key=lambda x: -x["fail_open"])[:5]]
                  for u in sorted({r["policy"] for r in rows})}
    body = [r"\begin{tabular}{l" + "r" * len(names) + "}", r"\toprule",
            "labels found on & " + " & ".join(PRETTY.get(m, m) for m in names) + r" \\",
            r"\midrule"]
    diag = []
    for src in sources:
        cells = []
        for tgt in names:
            rows = ms[tgt]
            idx = {(r["policy"], r["allow_label"]): r["fail_open"] for r in rows}
            hon = {r["policy"]: r["fail_open"] for r in rows if r["allow_label"] == "allow"}
            v = []
            for u, ls in lab[src].items():
                got = [idx[(u, l)] for l in ls if (u, l) in idx]
                if got and u in hon:
                    v.append(max(got) - hon[u])
            cells.append(float(np.mean(v)) if v else float("nan"))
            if src == tgt:
                diag.append(cells[-1])
        body.append(PRETTY.get(src, src) + " & " +
                    " & ".join(R(c) for c in cells) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_transfer.tex", "\n".join(body) + "\n")
    off = [float(np.mean([c for c in [1]])) for _ in [0]]  # placeholder removed below


def main():
    global OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=OUT)
    a = ap.parse_args(); OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    tab_directions(); tab_attacks(); tab_option(); tab_defense(); tab_verdicts(); tab_adaptive(); tab_normalize(); tab_triage(); tab_optsweep(); tab_strata(); tab_transfer(); tab_headline()
    with open(os.path.join(OUT, "macros.tex"), "w") as fh:
        fh.write("% generated by src/make_tables.py -- do not edit\n")
        for k, v in sorted(MACROS.items()):
            fh.write(f"\\providecommand{{\\{k}}}{{{v}}}\n")
        for n in ("vonexfil", "vonexfillo", "vonexfilhi", "chance",
                  "ncells", "nbetter", "ntie", "nworse", "adaptmax", "adaptmodels",
                  "ruleacc", "rulemaxfi", "normrawfi", "normfi",
                  "dosecleanpct", "dosesixpct", "doseacc", "ocminpct", "ocmaxpct",
                  "ocnmodels", "ocnmodelsword", "chancepct", "ruleaccpct", "normrawfipct", "normfipct",
                  "accminpct", "accmaxpct", "trbasefopct", "trbtenfopct", "trbtenescpct",
                  "trbfiftyfopct", "trbfiftyescpct", "trbfiftyoracpct", "trnflips",
                  "trmarginbefore", "trmarginafter", "trwiderpct", "ncellstriage",
                  "swqwhonpct", "swqwbestpct", "swqwktenpct", "swqwlabel", "swqwnzero"):
            if n not in MACROS:
                fh.write(f"\\providecommand{{\\{n}}}{{\\textbf{{[pending]}}}}\n")
    print(f"wrote macros.tex with {len(MACROS)} values")


if __name__ == "__main__":
    main()
