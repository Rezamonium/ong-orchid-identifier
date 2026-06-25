"""
24_pipeline_figure.py — Two-stage identification pipeline schematic (manuscript Figure 1).

Clean horizontal flowchart: Stage 1 (genus classification, blue swimlane) feeds an
open-set distance gate and Stage 2 (species retrieval, green swimlane). Vector + raster.

Run:  python notebooks/24_pipeline_figure.py
Out:  colab/result/eval/_comparison/figures/fig_pipeline.{png,pdf}
"""
import sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

FIG = Path(r"E:/Claude Code/ONG_v3/colab/result/eval/_comparison/figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})

# colours
BLUE_BAND, BLUE_EDGE = "#E8F0FE", "#1A56DB"
GREEN_BAND, GREEN_EDGE = "#E7F5EC", "#137333"
GREY_FILL, GREY_EDGE = "#F1F3F4", "#5F6368"
PURP_FILL, PURP_EDGE = "#EDE7F6", "#6A3FB5"
AMBER_FILL, AMBER_EDGE = "#FFF4E5", "#E8920C"
RED_FILL, RED_EDGE = "#FCE8E6", "#D93025"

fig, ax = plt.subplots(figsize=(13, 6.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def band(x0, x1, y0, y1, fill, edge, label):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle="round,pad=0.4,rounding_size=2.5",
                 linewidth=1.2, edgecolor=edge, facecolor=fill, alpha=0.55, zorder=0))
    ax.text(x0 + 1.5, y1 - 2.6, label, ha="left", va="top",
            fontsize=10.5, fontweight="bold", color=edge, zorder=1)


def box(cx, cy, w, h, text, fill, edge, fs=9.5, fw="normal"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.3,rounding_size=1.6",
                 linewidth=1.5, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight=fw, color="#202124", zorder=3, linespacing=1.25)
    return cx, cy, w, h


def diamond(cx, cy, w, h, text, fill, edge):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=1.5,
                 edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=9, zorder=3, linespacing=1.2)
    return cx, cy, w, h


def arrow(p0, p1, color="#202124", style="-", lw=1.6, rad=0.0, label=None, lpos=0.5, ldy=2.2):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, lw=lw,
                        color=color, linestyle=style, zorder=1,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if label:
        mx, my = p0[0] + (p1[0] - p0[0]) * lpos, p0[1] + (p1[1] - p0[1]) * lpos
        ax.text(mx, my + ldy, label, ha="center", va="bottom", fontsize=8.2,
                color=color, fontweight="bold", zorder=4)


# ── swimlanes ──────────────────────────────────────────────────────────────
band(20, 99.5, 66, 97, BLUE_BAND, BLUE_EDGE, "Stage 1 — Genus classification")
band(20, 99.5, 3, 33, GREEN_BAND, GREEN_EDGE, "Stage 2 — Species retrieval")

# ── input ──────────────────────────────────────────────────────────────────
inp = box(9, 50, 14, 16, "User photo(s)\nof orchid\n(1 or more)", GREY_FILL, GREY_EDGE, fw="bold")

# ── Stage 1 boxes ──────────────────────────────────────────────────────────
bb = box(34, 80, 21, 15, "DINOv2 ViT-L/14\nbackbone\n(fine-tuned, 448 px)", "white", BLUE_EDGE)
hd = box(60, 80, 20, 15, "Linear head\n(120 genera) +\ntemperature scaling", "white", BLUE_EDGE)
gn = box(86, 80, 20, 15, "Predicted genus\n+ calibrated\nconfidence", "white", BLUE_EDGE, fw="bold")

# ── embedding + open-set gate ──────────────────────────────────────────────
emb = box(34, 50, 20, 13, "1024-d embedding\n(L2-normalised)", PURP_FILL, PURP_EDGE)
gate = diamond(60, 50, 22, 22, "Distance to\nnearest reference\n> threshold τ ?", AMBER_FILL, AMBER_EDGE)
nov = box(89, 50, 19, 16, "Flag as novel genus\n→ expert review\n(multi-access key)", RED_FILL, RED_EDGE, fw="bold")

# ── Stage 2 boxes ──────────────────────────────────────────────────────────
fa = box(40, 18, 23, 15, "FAISS index\n(restricted to\npredicted genus)", "white", GREEN_EDGE)
tk = box(67, 18, 19, 15, "Top-k nearest\nreferences +\nspecies labels", "white", GREEN_EDGE)
sg = box(90, 18, 18, 16, "Ranked species\nsuggestions →\nuser determination", "white", GREEN_EDGE, fw="bold")

# ── arrows: Stage 1 ────────────────────────────────────────────────────────
arrow((16, 53), (23.5, 78), color=BLUE_EDGE)            # input -> backbone
arrow((44.5, 80), (50, 80), color=BLUE_EDGE)            # backbone -> head
arrow((70, 80), (76, 80), color=BLUE_EDGE)              # head -> genus

# backbone -> embedding -> gate
arrow((34, 72.5), (34, 56.5), color=PURP_EDGE)          # backbone -> embedding
arrow((44, 50), (49, 50), color=PURP_EDGE)              # embedding -> gate

# gate YES -> novel
arrow((71, 50), (79.5, 50), color=RED_EDGE, label="yes")
# gate NO -> Stage 2 FAISS
arrow((60, 39), (40, 25.5), color=GREEN_EDGE, label="no", lpos=0.35, ldy=1.4)
# predicted genus -> FAISS: dashed arrow = the predicted genus restricts the index
# (explained in the figure caption; FAISS box text also states the restriction)
arrow((86, 72.5), (42, 25.8), color="#5F6368", style=(0, (4, 3)), lw=1.3, rad=-0.32)

# Stage 2 flow
arrow((51.5, 18), (57.5, 18), color=GREEN_EDGE)         # faiss -> topk
arrow((76.5, 18), (81, 18), color=GREEN_EDGE)           # topk -> suggestions

fig.tight_layout(pad=0.5)
FIG.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig_pipeline.{ext}", dpi=300, bbox_inches="tight")
print("Saved", FIG / "fig_pipeline.png")
