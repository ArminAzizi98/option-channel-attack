"""Figures for the guardrail security paper.

Categorical hues are the reference palette's slots in fixed order, never cycled. Scatters
use a single series with direct labels, which keeps identity off colour entirely and so
stays inside the all-pairs series cap.
"""
import os, sys, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from guardlab.config import ROOT

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8982", "#ffffff"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.family": "serif", "font.size": 9.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.titlesize": 10,
    "axes.labelsize": 9.5, "legend.fontsize": 8.5, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
})
OUT = f"{ROOT}/figs"


def pct_axis(ax, which="y", hi=1.0, step=0.2):
    """Label a rate axis in percent, so figures match the percentages used in the text."""
    import numpy as _np
    ticks = _np.arange(0, hi + 1e-9, step)
    labs = [f"{100*t:.0f}%" for t in ticks]
    if which in ("y", "both"):
        ax.set_yticks(ticks); ax.set_yticklabels(labs)
    if which in ("x", "both"):
        ax.set_xticks(ticks); ax.set_xticklabels(labs)


def style(ax, xlabel, ylabel, title=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#e8e7e2", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", pad=8)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p + ".pdf"); fig.savefig(p + ".png")
    plt.close(fig)
    print("wrote", p + ".pdf")


# ---------------------------------------------------------------- fig 1
def fig_directions():
    """Both error rates at once. Colour is the model family, shape is the task.

    Three families keeps the categorical palette inside the all-pairs limit, and the task
    is carried by marker shape rather than a fourth hue.
    """
    import glob as _g
    fam_of = {"laya-td": "laya", "laya-en": "laya", "laya-ml": "laya",
              "von": "von", "rlcd": "rlcd",
              "qwen1.5b": "qwen", "qwen7b": "qwen"}
    pts = []
    for f in sorted(_g.glob(f"{ROOT}/runs/real/"
                            "*__*__benign_suffix_x3.npz")):
        m, t, _a = os.path.basename(f)[:-4].split("__")
        if m not in fam_of:
            continue
        z = np.load(f)
        c, g = z["clean"], z["gold"]
        pred = c.argmax(1)
        fo = float((pred[g == 1] == 0).mean())
        fc = float((pred[g == 0] == 1).mean())
        pts.append((fam_of[m], m, t, fc, fo))

    fam_color = {"laya": SERIES[0], "von": SERIES[1], "qwen": SERIES[2],
                 "rlcd": SERIES[3]}
    task_mark = {"injection": "o", "jailbreak": "s", "toxic": "^"}

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    # the unsafe half of the plot
    ax.axhspan(0.5, 1.06, color="#f6ece8", zorder=0)
    ax.annotate("allows most prohibited actions", (0.52, 1.015), fontsize=8.5,
                color="#9a5a42", ha="center", va="top", zorder=1)
    ax.plot([0, 1], [1, 0], color=MUTED, linewidth=1.0, linestyle=(0, (5, 4)), zorder=2)
    ax.annotate("a gate that ignores its input", (0.60, 0.455), fontsize=8,
                color=MUTED, rotation=-38, ha="center", va="bottom", zorder=3)

    for fam, model, task, x, y in pts:
        ax.scatter([x], [y], s=72, marker=task_mark[task], color=fam_color[fam],
                   zorder=5, linewidth=0, alpha=0.95)
    ax.scatter([0], [0], marker="*", s=240, color=INK, zorder=6)
    ax.annotate("a useful gate\nwould be here", (0, 0), fontsize=9, color=INK,
                xytext=(14, 10), textcoords="offset points", zorder=6)
    # placed in the empty band left of the bottom-right cluster; at the corner it sat on
    # top of four markers
    ax.annotate("blocks most\npermitted actions", (0.76, 0.085), fontsize=8.5, color=INK2,
                ha="center", va="center", zorder=6)

    from matplotlib.lines import Line2D
    fams = [f for f in ("laya", "von", "rlcd", "qwen")
            if any(p[0] == f for p in pts)]
    h1 = [Line2D([], [], marker="o", linestyle="", markersize=8, color=fam_color[f], label=f)
          for f in fams]
    h2 = [Line2D([], [], marker=task_mark[t], linestyle="", markersize=8, color=INK2, label=t)
          for t in ("injection", "jailbreak", "toxic")]
    lg1 = ax.legend(handles=h1, title="model family", loc="upper right",
                    fontsize=8.5, title_fontsize=8.5, handletextpad=0.3)
    ax.add_artist(lg1)
    ax.legend(handles=h2, title="screening task", loc="center right",
              fontsize=8.5, title_fontsize=8.5, handletextpad=0.3)

    style(ax, "fail-closed rate: permitted actions the gate blocked",
          "fail-open rate: prohibited actions the gate allowed")
    ax.set_title("Each gate is unsafe in one direction or the other",
                 loc="left", pad=10, fontsize=10)
    ax.set_xlim(-0.05, 1.08); ax.set_ylim(-0.05, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    save(fig, "fig1_error_directions")


# ---------------------------------------------------------------- fig 2
def fig_dose():
    """Attack strength against a gate that works on clean input. Read from saved cells."""
    import glob
    pol, model = "unauth_irrev", "laya-td"
    def cell(atk):
        f = f"{ROOT}/runs/attack/{model}__{pol}__{atk}.npz"
        if not os.path.exists(f):
            return None
        z = np.load(f); c, a, g = z["clean"], z["adv"], z["gold"]
        cb = (g == 1) & (c.argmax(1) == 1)
        clean_fo = float((c.argmax(1)[g == 1] == 0).mean())
        return clean_fo, float((a.argmax(1)[cb] == 0).mean())
    d1, d3, d6 = cell("distraction_k=1"), cell("distraction_k=3"), cell("distraction_k=6")
    p1, p3 = cell("persuasion_k=1"), cell("persuasion_k=3")
    if d1 is None:
        return
    clean_fo = d1[0]
    k_d = [0, 1, 3, 6]
    distract = [clean_fo, d1[1], d3[1], d6[1]]
    k_p = [0, 1, 3]
    persuade = [clean_fo, p1[1], p3[1]]

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(k_d, distract, marker="o", markersize=4.5, linewidth=2, color=SERIES[0],
            label="routine server log lines", zorder=3)
    ax.plot(k_p, persuade, marker="s", markersize=4.5, linewidth=2, color=SERIES[1],
            label="unverified approval notes", zorder=3)
    for x, y in list(zip(k_d, distract))[1:]:
        # the k=1 label sits under the steepest part of the line, so place it to the right
        off, ha = ((12, -7), "left") if x == 1 else ((0, -14), "center")
        ax.annotate(f"{100*y:.0f}%", (x, y), fontsize=8.5, color=INK2,
                    xytext=off, textcoords="offset points", ha=ha)
    for x, y in list(zip(k_p, persuade))[1:]:
        ax.annotate(f"{100*y:.0f}%", (x, y), fontsize=8.5, color=INK2,
                    xytext=(0, 8), textcoords="offset points", ha="center")
    # anchored in the empty band under both curves; the previous placement ran the text
    # straight through the 33% data label
    ax.annotate("clean gate:\nalmost never\nfails open", xy=(0, clean_fo), fontsize=8.5,
                color=INK2, xytext=(1.45, 0.075),
                textcoords="data", ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5,
                                shrinkA=2, shrinkB=3))
    style(ax, "lines of attacker-controlled text appended to the trace", "fail-open rate",
          "Text that says nothing about the policy still opens the gate")
    ax.legend(loc="lower right", handlelength=1.8)
    ax.set_ylim(-0.04, 0.95)
    pct_axis(ax, "y", hi=0.8, step=0.2)
    save(fig, "fig2_dose_response")


# ---------------------------------------------------------------- fig 3
def fig_option_channel():
    """Renaming one option, touching no trace text."""
    f = f"{ROOT}/runs/attack/laya-td__positive.json"
    if not os.path.exists(f):
        return
    rows = None
    for r in json.load(open(f)):
        if r.get("attack") == "option_channel" and r.get("policy") == "unauth_irrev":
            rows = r["all_rows"]
    if not rows:
        return
    # keep the honest naming in the plot even when it is not among the worst cases
    honest_row = [x for x in rows
                  if x["block_label"] == "block" and x["allow_label"] == "allow"]
    top = sorted(rows, key=lambda x: x["fail_open"])[-11:]
    if honest_row and honest_row[0] not in top:
        top = [honest_row[0]] + top
    rows = sorted(top, key=lambda x: x["fail_open"])
    labels = [f"{x['block_label']} / {x['allow_label']}" for x in rows]
    vals = [x["fail_open"] for x in rows]
    honest = [i for i, x in enumerate(rows)
              if x["block_label"] == "block" and x["allow_label"] == "allow"]
    cols = [SERIES[1] if i in honest else SERIES[0] for i in range(len(rows))]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=cols, linewidth=0, zorder=3, height=0.72)
    for yi, v in zip(y, vals):
        ax.annotate(f"{100*v:.0f}%", (v, yi), fontsize=8.5, color=INK2,
                    xytext=(4, 0), textcoords="offset points", va="center")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
    style(ax, "fail-open rate", "labels installed  (block option / allow option)",
          "Two label words open the gate completely; the rest do nothing")
    ax.set_xlim(0, 1.16)
    pct_axis(ax, "x", hi=1.0, step=0.25)
    # the orange bar carries identity; the text stays in ink
    ax.annotate("honest naming", (vals[honest[0]], honest[0]), fontsize=6.5,
                color=INK2, xytext=(26, -1), textcoords="offset points", va="center")
    save(fig, "fig3_option_channel")


# ---------------------------------------------------------------- fig 4
def fig_margin():
    """Robustness tracks the clean decision margin. Read from the full-size real-task cells."""
    f = f"{ROOT}/runs/real/laya-td__jailbreak__benign_suffix_x3.npz"
    if not os.path.exists(f):
        return
    z = np.load(f); c, a, g = z["clean"], z["adv"], z["gold"]
    cb = (g == 1) & (c.argmax(1) == 1)
    margin = np.abs(c[:, 1] - 0.5)
    edges = [(0.00, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01)]
    bands, flip, ns = [], [], []
    for lo, hi in edges:
        sel = cb & (margin >= lo) & (margin < hi)
        if sel.sum() == 0:
            continue
        bands.append(f"[{lo:.2f},\n{hi:.2f})" if hi <= 1 else f"$\\geq${lo:.2f}")
        flip.append(float((a.argmax(1)[sel] == 0).mean()))
        ns.append(int(sel.sum()))
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    x = np.arange(len(bands))
    ax.bar(x, flip, width=0.62, color=SERIES[0], linewidth=0, zorder=3)
    for xi, v, ni in zip(x, flip, ns):
        ax.annotate(f"{100*v:.0f}%\nn={ni}", (xi, v), fontsize=8.5, color=INK2,
                    xytext=(0, 4), textcoords="offset points", ha="center")
    ax.set_xticks(x); ax.set_xticklabels(bands, fontsize=8.5)
    style(ax, "how far the clean decision was from the 0.5 boundary",
          "correct blocks the attack reversed",
          "Decisions made with less margin are the ones that flip")
    ax.set_ylim(0, max(flip) * 1.40 if flip else 1)
    pct_axis(ax, "y", hi=max(0.8, round(max(flip) + 0.1, 1)), step=0.2)
    save(fig, "fig4_margin")


# ---------------------------------------------------------------- fig 5
def fig_tradeoff():
    """Defenses against the control: the undefended gate biased toward blocking."""
    f = f"{ROOT}/runs/tradeoff/laya-td__unauth_irrev.json"
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    pts = [(c["clean_fc"], c["attacked_fo"]) for c in d["curve"]]
    seen, curve = set(), []
    for fc, fo in pts:                      # the sweep saturates; keep the distinct points
        if fc in seen:
            continue
        seen.add(fc); curve.append((fc, fo))
    curve_fc = [x[0] for x in curve]
    curve_fo = [x[1] for x in curve]
    short = {"D1 canonical labels": "D1 canonical labels",
             "D3 ensemble of 3": "D3 ensemble",
             "D4+D6 disagree,failclosed": "D4+D6 disagree",
             "D5 field-structured": "D5 field-structured"}
    defs = [(short.get(r["defense"], r["defense"]), r["clean_fc"], r["attacked_fo"])
            for r in d["defenses"]]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(curve_fc, curve_fo, marker="o", markersize=4.5, linewidth=2,
            color=SERIES[1], label="undefended, threshold swept", zorder=3)
    ax.scatter([d[1] for d in defs], [d[2] for d in defs], s=44, color=SERIES[0],
               zorder=4, linewidth=0, label="defenses")
    # Three defenses land within a few points of each other, so direct labels collide.
    # The labels go in a column in the empty left region, ordered by their point's x so
    # the leader lines do not cross, with each line drawn back to its own point.
    for i, (nm, x, y) in enumerate(sorted(defs, key=lambda d: d[1])):
        ax.annotate(nm, (x, y), fontsize=8, color=INK2,
                    xytext=(0.015, 0.34 - 0.085 * i), textcoords="data",
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5,
                                    shrinkA=2, shrinkB=3))
    style(ax, "permitted actions blocked", "prohibited actions allowed under attack",
          "Defenses reach operating points that thresholding cannot")
    ax.set_xlim(-0.04, 1.06); ax.set_ylim(-0.05, 0.86)
    pct_axis(ax, "x", hi=1.0, step=0.25); pct_axis(ax, "y", hi=0.8, step=0.2)
    ax.legend(loc="upper right", handlelength=1.8)
    save(fig, "fig5_tradeoff")


# ---------------------------------------------------------------- fig 6
def fig_triage():
    """Why escalating low-confidence cases does not catch attacks."""
    fc = f"{ROOT}/runs/triage/triage.json"
    fm = f"{ROOT}/runs/triage/margins.json"
    if not (os.path.exists(fc) and os.path.exists(fm)):
        return
    cells, mar = json.load(open(fc)), json.load(open(fm))
    f, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))

    # (a) margin before vs after, on the decisions a successful attack reversed
    ax = axes[0]
    per = [c for c in mar["per_cell"] if c["n"] >= 5]
    xb = [c["margin_before"] for c in per]
    ya = [c["margin_after"] for c in per]
    sz = [14 + 6.5 * np.sqrt(c["n"]) for c in per]
    ax.plot([0, 0.5], [0, 0.5], color=INK2, lw=0.9, ls="--", zorder=1)
    ax.scatter(xb, ya, s=sz, facecolor=SERIES[0], edgecolor="white", linewidth=0.7,
               alpha=0.85, zorder=3)
    # below the diagonal in the empty lower-right corner; on the line it covered the markers
    ax.text(0.475, 0.055, "on this line, equally confident\nbefore and after the attack",
            fontsize=8, color=INK2, ha="right", va="bottom")
    ax.annotate("", xy=(0.425, 0.425), xytext=(0.455, 0.105),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    q = mar["pooled"]
    ax.text(0.025, 0.47, f"{q['n_flips']} reversed decisions, pooled margin\n"
                         f"{q['margin_before']:.3f} before and {q['margin_after']:.3f} after; "
                         f"wider\nafter the attack in {100*q['frac_wider']:.0f}% of them",
            fontsize=8.2, color=INK, va="top")
    style(ax, "margin before the attack", "margin after the attack",
          "(a) A reversed decision is no less confident")
    ax.set_xlim(0, 0.5); ax.set_ylim(0, 0.5)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5]); ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])

    # (b) escalate-by-margin against an oracle given the same review budget
    ax = axes[1]
    budgets = sorted({r["budget"] for c in cells for r in c["rows"]})
    esc, fo = [], []
    for b in budgets:
        rs = [r for c in cells for r in c["rows"] if r["budget"] == b]
        esc.append(float(np.mean([r["esc_attacked"] for r in rs])))
        fo.append(float(np.nanmean([r["fo_attacked"] for r in rs])))
    share = fo[0] * 0.5
    # drawn on a fine grid: the oracle reaches zero at `share`, not at the last measured point
    gx = np.linspace(0, 0.53, 300)
    gy = [max(0.0, 1.0 - e / share) * fo[0] for e in gx]
    ax.plot(gx, gy, color=INK2, lw=1.1, ls="--", zorder=2)
    orac = [max(0.0, 1.0 - e / share) * fo[0] for e in esc]
    ax.plot(esc, fo, color=SERIES[1], lw=1.8, marker="o", ms=5.5,
            markerfacecolor="white", markeredgewidth=1.5, zorder=3)
    ax.text(esc[-1] - 0.01, fo[-1] - 0.045, "escalate the\nlowest-margin items",
            fontsize=8.4, color=SERIES[1], ha="right", va="top")
    # in the empty wedge below the oracle line; across it the text was unreadable, and to
    # the right it ran into the orange label
    ax.annotate("oracle spending the same\nreview budget on the items\nthat do fail open",
                xy=(0.205, 0.195), xytext=(0.018, 0.052), fontsize=8.2, color=INK2,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.7))
    style(ax, "share of traffic sent for review", "fail-open rate under attack",
          "(b) The review budget buys little safety")
    pct_axis(ax, "both", hi=0.6, step=0.1)
    ax.set_xlim(-0.012, 0.53); ax.set_ylim(0, 0.66)

    f.tight_layout()
    save(f, "fig6_triage")


# ---------------------------------------------------------------- fig 7
def fig_optsweep():
    """Which labels open the gate, and how many tries an attacker needs."""
    import glob as _g
    ms = {}
    for f in sorted(_g.glob(f"{ROOT}/runs/optsweep/*.json")):
        j = json.load(open(f)); ms[j["model"]] = j["rows"]
    if "laya-td" not in ms:
        return
    f, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

    # (a) kinds of label, on the one policy where the clean gate is accurate
    ax = axes[0]
    rows = [r for r in ms["laya-td"] if r["policy"] == "unauth_irrev"]
    hon = [r["fail_open"] for r in rows if r["allow_label"] == "allow"][0]
    nice = {"security_action": "names a security\naction", "neutral_syntactic": "neutral token",
            "control_unrelated": "unrelated noun", "allow_synonym": "synonym of allow",
            "severity": "names a severity", "block_synonym": "synonym of block"}
    st = sorted({r["stratum"] for r in rows},
                key=lambda t: np.mean([r["fail_open"] for r in rows if r["stratum"] == t]))
    means = [np.mean([r["fail_open"] for r in rows if r["stratum"] == t]) for t in st]
    bars = ax.barh(range(len(st)), means, height=0.62,
                   color=[SERIES[2] if m < 0.05 else SERIES[1] for m in means],
                   edgecolor="white", linewidth=0.8)
    for i, (t, m) in enumerate(zip(st, means)):
        v = [r["fail_open"] for r in rows if r["stratum"] == t]
        ax.plot([max(v)], [i], marker="|", ms=11, mew=1.6, color=INK2)
        ax.text(m + 0.015, i, f"{100*m:.0f}%", va="center", fontsize=8.2, color=INK)
    ax.axvline(hon, color=INK, lw=1.1, ls=":")
    ax.set_yticks(range(len(st))); ax.set_yticklabels([nice.get(t, t) for t in st], fontsize=8.5)
    style(ax, "fail-open rate", None, "(a) Which kinds of label open this gate")
    pct_axis(ax, "x", hi=1.0, step=0.25)
    ax.set_xlim(0, 1.04)
    # stacked under the bars; side by side they overlapped, and above the bars the first
    # note sat on the panel title
    ax.text(0.022, -1.05, f"dotted line: honest naming of this option, {100*hon:.0f}%\n"
                        f"|  best label in the group",
            fontsize=8, color=INK2, ha="left", va="top", linespacing=1.5)
    ax.set_ylim(-2.1, len(st) - 0.35)

    # (b) expected best fail-open against the attacker's query budget
    ax = axes[1]
    ks = [1, 2, 5, 10, 20, 50, 100, 198]
    rng = np.random.default_rng(0)
    # six models and a five-colour palette collided, so the label-invariant pair gets its
    # own flat grey style and the four attacked models keep distinct colours
    order = sorted(ms, key=lambda m: m in ("von", "rlcd"))
    ci = 0
    for m in order:
        rws = ms[m]
        pols = sorted({r["policy"] for r in rws})
        ys = []
        for k in ks:
            v = []
            for u in pols:
                a = np.array([r["fail_open"] for r in rws if r["policy"] == u])
                if len(a) < k:
                    continue
                idx = np.argsort(rng.random((400, len(a))), axis=1)[:, :k]
                v.append(float(a[idx].max(axis=1).mean()))
            ys.append(float(np.mean(v)))
        flat = max(ys) - min(ys) < 1e-9
        if flat:
            ax.plot(ks, ys, lw=1.6, ls=(0, (4, 3)), color=MUTED, zorder=2)
            ax.annotate(m, (ks[-1], ys[-1]), xytext=(-4, 7), textcoords="offset points",
                        fontsize=8.2, color=MUTED, ha="right")
        else:
            # the four curves converge near 100%, so they are named in a legend rather
            # than directly at the line ends, where the labels overlapped
            ax.plot(ks, ys, lw=1.8, marker="o", ms=4.2, color=SERIES[ci % len(SERIES)],
                    markerfacecolor="white", markeredgewidth=1.3, label=m, zorder=3)
            ci += 1
    style(ax, "labels tried (random order)", "fail-open rate reached",
          "(b) What the attack costs the attacker")
    ax.set_xscale("log"); ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=8.5)
    pct_axis(ax, "y", hi=1.0, step=0.2)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper left", ncol=2, fontsize=8.2, handlelength=1.5,
              columnspacing=1.1, borderpad=0.2, labelspacing=0.3)
    ax.text(1.1, 0.045, "dashed: label-invariant, flat at every budget",
            fontsize=8, color=MUTED)

    f.tight_layout()
    save(f, "fig7_optsweep")


if __name__ == "__main__":
    fig_directions(); fig_dose(); fig_option_channel(); fig_margin(); fig_tradeoff(); fig_triage(); fig_optsweep()
