"""Schematic: what a typed decision model is, and where the two attack channels enter."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from guardlab.config import ROOT

SER = ["#2a78d6", "#eb6834"]            # definition, label
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#8a8982", "#ffffff"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.family": "serif",
                     "savefig.bbox": "tight", "savefig.facecolor": SURF})

W, H = 7.0, 3.1
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off"); ax.set_facecolor(SURF)
MONO = 6.3
CW = 100.0 / W * (0.655 * MONO / 72.0)   # monospace advance width in data units


def panel(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.0",
                                linewidth=0.9, edgecolor=MUTED, facecolor="#ffffff", zorder=2))
    ax.text(x + 1.8, y + h - 2.6, title, ha="left", va="center", fontsize=7.4,
            color=INK, fontweight="bold", zorder=3)


def mono(x, y, parts):
    """parts: list of (text, colour); laid out left to right in monospace.

    Each segment starts at the measured right edge of the one before it. An estimated
    character width was used here before and ran the segments into each other.
    """
    cx = x
    for txt, col in parts:
        t = ax.text(cx, y, txt, ha="left", va="center", fontsize=MONO, color=col,
                    family="monospace", zorder=3)
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        cx = ax.transData.inverted().transform((bb.x1, bb.y0))[0]


# ---- state -------------------------------------------------------------
panel(2, 29, 40, 18, "state: what is being judged")
mono(4.0, 40.0, [("destination: paste.ee", INK2)])
mono(4.0, 36.0, [("body: AKIA3F9XQ2LM...", INK2)])
mono(4.0, 32.0, [("tool output: 200 OK ...", INK2)])

# ---- question ----------------------------------------------------------
panel(2, 3, 40, 22, "typed question: what to decide")
mono(4.0, 18.5, [("instructions: apply the policy", INK2)])
mono(4.0, 13.0, [("block: ", SER[1]), ("host is outside the company", SER[0])])
mono(4.0, 8.5, [("allow: ", SER[1]), ("host is internal", SER[0])])
ax.text(4.0, 5.2, "orange: option label        blue: option definition",
        fontsize=5.9, color=MUTED, ha="left", va="center")

# ---- model -------------------------------------------------------------
ax.add_patch(FancyBboxPatch((52, 19), 18, 13, boxstyle="round,pad=0.5,rounding_size=1.0",
                            linewidth=1.1, edgecolor=INK, facecolor="#f4f4f2", zorder=2))
ax.text(61, 28.0, "typed decision", ha="center", fontsize=7.6, color=INK, fontweight="bold")
ax.text(61, 24.6, "model", ha="center", fontsize=7.6, color=INK, fontweight="bold")
ax.text(61, 21.4, "generates no text", ha="center", fontsize=5.9, color=MUTED)

# ---- output ------------------------------------------------------------
panel(78, 19, 20, 13, "probabilities")
mono(80.0, 24.5, [("block  0.57", INK2)])
mono(80.0, 21.0, [("allow  0.43", INK2)])

for a, b in [((43.0, 36), (51.2, 28)), ((43.0, 13), (51.2, 23)), ((70.8, 25.5), (77.2, 25.5))]:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8,
                                 color=MUTED, linewidth=0.9, zorder=1))

# ---- attack channels ---------------------------------------------------
ax.add_patch(FancyArrowPatch((22, 50.5), (22, 47.6), arrowstyle="-|>", mutation_scale=8,
                             color=SER[0], linewidth=1.3, zorder=4))
ax.text(24.5, 50.0, "channel 1: the attacker writes part of the state",
        fontsize=6.8, color=INK, va="center", ha="left")
ax.add_patch(FancyArrowPatch((22, -1.2), (22, 2.0), arrowstyle="-|>", mutation_scale=8,
                             color=SER[1], linewidth=1.3, zorder=4))
ax.text(24.5, -1.6, "channel 2: the attacker sets an option label",
        fontsize=6.8, color=INK, va="center", ha="left")
ax.set_ylim(-4, 54)

out = f"{ROOT}/figs/fig0_overview"
fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
print("wrote", out + ".pdf")
