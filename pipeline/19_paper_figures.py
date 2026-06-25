"""
19_paper_figures.py — Generate the §3.3 paper figures from the clean-pilot eval outputs.

Figures (→ colab/result/eval/_comparison/figures/):
  fig_macro_ci.png/.pdf        §3.3.1  macro top-1 ±95% CI per backbone (ViT vs CNN) + ECE panel
  fig_dinov2_per_genus.png     §3.3.2  DINOv2 per-genus top-1, sorted, bars coloured by test support
  fig_dinov2_training.png      §3.3.2  DINOv2 training curves (loss + val metrics, two-phase)
  fig_dinov2_confusion.png     §3.3.2  DINOv2 row-normalised confusion matrix

Stdlib + numpy + matplotlib only. Run:
  python notebooks/19_paper_figures.py
"""

import csv
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ONG = Path(r"E:/Claude Code/ONG_v3")
EVAL = ONG / "colab/result/eval"
OUT = EVAL / "_comparison/figures"
OUT.mkdir(parents=True, exist_ok=True)

# (key, display, family) ordered for loading; plots re-sort by macro top-1
MODELS = [
    ("dinov2l",     "DINOv2\nViT-L/14",      "ViT"),
    ("bioclip2",    "BioCLIP-2\nViT-L/14",   "ViT"),
    ("convnextv2l", "ConvNeXt-V2-L",         "CNN"),
    ("effnetv2l",   "EfficientNetV2-L",      "CNN"),
]
FAMILY_COLOR = {"ViT": "#2c6fbb", "CNN": "#e07b39"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


# ── Figure 1: macro top-1 ±95% CI + ECE ──────────────────────────────────────────
def fig_macro_ci():
    rows = []
    for key, disp, fam in MODELS:
        with open(EVAL / f"{key}_global/results.json", encoding="utf-8") as f:
            c = json.load(f)["classification"]
        lo, hi = c["macro_top1_ci95"]
        rows.append(dict(disp=disp, fam=fam, macro=c["macro_top1"] * 100,
                         lo=lo * 100, hi=hi * 100, ece=c["ece"]))
    rows.sort(key=lambda r: r["macro"], reverse=True)
    x = np.arange(len(rows))
    colors = [FAMILY_COLOR[r["fam"]] for r in rows]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.6, 1]})

    macro = [r["macro"] for r in rows]
    yerr = np.array([[r["macro"] - r["lo"] for r in rows],
                     [r["hi"] - r["macro"] for r in rows]])
    a1.bar(x, macro, color=colors, width=0.62, zorder=2)
    a1.errorbar(x, macro, yerr=yerr, fmt="none", ecolor="#333", capsize=5, lw=1.4, zorder=3)
    for xi, r in zip(x, rows):
        a1.text(xi, r["hi"] + 1.0, f"{r['macro']:.1f}", ha="center", va="bottom", fontsize=10)
    a1.set_ylabel("Macro top-1 accuracy (%)")
    a1.set_title("Genus classification — macro top-1 (95% CI)")
    a1.set_xticks(x); a1.set_xticklabels([r["disp"] for r in rows], fontsize=8)
    a1.set_ylim(0, max(r["hi"] for r in rows) + 7)
    a1.grid(axis="y", ls=":", alpha=0.5)

    a2.bar(x, [r["ece"] for r in rows], color=colors, width=0.62, zorder=2)
    for xi, r in zip(x, rows):
        a2.text(xi, r["ece"] + 0.004, f"{r['ece']:.3f}", ha="center", va="bottom", fontsize=9)
    a2.set_ylabel("Expected Calibration Error (lower = better)")
    a2.set_title("Calibration (ECE)")
    a2.set_xticks(x); a2.set_xticklabels([r["disp"] for r in rows], fontsize=8)
    a2.set_ylim(0, max(r["ece"] for r in rows) * 1.25)
    a2.grid(axis="y", ls=":", alpha=0.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f]) for f in ("ViT", "CNN")]
    a1.legend(handles, ["Vision Transformer", "Convolutional"], frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "fig_macro_ci")


# ── Figure 2: DINOv2 per-genus top-1, coloured by support ─────────────────────────
def fig_per_genus():
    rows = list(csv.DictReader(open(EVAL / "dinov2l_global/per_genus_accuracy.csv", encoding="utf-8")))
    rows = [(r["genus"], int(r["support"]), float(r["top1_acc"]) * 100) for r in rows]
    rows.sort(key=lambda r: r[2], reverse=True)
    genera = [r[0] for r in rows]
    support = np.array([r[1] for r in rows])
    acc = [r[2] for r in rows]
    x = np.arange(len(rows))

    cmap = plt.cm.viridis
    cvals = np.log10(support)
    colors = cmap((cvals - cvals.min()) / (cvals.max() - cvals.min()))

    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.bar(x, acc, color=colors, width=0.85)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_xlabel("Genus (sorted by accuracy)")
    ax.set_title("DINOv2 ViT-L/14 — per-genus top-1 accuracy, coloured by test support")
    ax.set_xticks(x); ax.set_xticklabels(genera, rotation=90, fontsize=6)
    ax.set_xlim(-0.7, len(rows) - 0.3); ax.set_ylim(0, 105)
    ax.grid(axis="y", ls=":", alpha=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=cvals.min(), vmax=cvals.max()))
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Test support (images)")
    ticks = [1, 5, 10, 50, 100, 500, 800]
    ticks = [t for t in ticks if support.min() <= t <= support.max()]
    cbar.set_ticks(np.log10(ticks)); cbar.set_ticklabels([str(t) for t in ticks])
    fig.tight_layout()
    save(fig, "fig_dinov2_per_genus")


# ── Figure 3: DINOv2 training curves ──────────────────────────────────────────────
def fig_training():
    p = next((EVAL / "dinov2l_global").glob("*folder stats*/training_history.csv"))
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    n_warm = sum(1 for r in rows if r["phase"] == "warmup")
    x = np.arange(1, len(rows) + 1)
    loss = [float(r["loss"]) for r in rows]
    macro = [float(r["val_macro"]) * 100 for r in rows]
    ema = [float(r["val_ema_macro"]) * 100 for r in rows]
    top1 = [float(r["val_top1"]) * 100 for r in rows]
    top5 = [float(r["val_top5"]) * 100 for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    handles = []
    handles += ax1.plot(x, loss, color="#999", lw=1.8, marker="o", ms=3, label="train loss")
    ax1.set_xlabel("Epoch (warm-up → fine-tune)")
    ax1.set_ylabel("Training loss", color="#666")
    ax1.tick_params(axis="y", labelcolor="#666")

    ax2 = ax1.twinx()
    handles += ax2.plot(x, top5, color="#7bb662", lw=1.8, label="val top-5")
    handles += ax2.plot(x, top1, color="#2c6fbb", lw=1.8, label="val top-1")
    handles += ax2.plot(x, macro, color="#d1495b", lw=1.8, label="val macro-F1")
    handles += ax2.plot(x, ema, color="#d1495b", lw=1.5, ls="--", label="val macro-F1 (EMA)")
    ax2.set_ylabel("Validation accuracy / macro-F1 (%)")
    ax2.set_ylim(0, 100)

    ax1.axvline(n_warm + 0.5, color="#bbb", ls=":", lw=1.4)
    ax1.text(n_warm + 0.5, ax1.get_ylim()[1], "  fine-tune →", va="top", fontsize=8, color="#888")
    ax1.set_title("DINOv2 ViT-L/14 — training dynamics")
    ax1.legend(handles, [h.get_label() for h in handles], loc="center right",
               fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, "fig_dinov2_training")


# ── Figure 4: DINOv2 row-normalised confusion matrix ──────────────────────────────
def fig_confusion():
    with open(EVAL / "dinov2l_global/confusion_matrix.csv", encoding="utf-8") as f:
        r = csv.reader(f)
        cols = next(r)[1:]
        labels, M = [], []
        for line in r:
            labels.append(line[0]); M.append([float(v) for v in line[1:]])
    M = np.array(M)
    Mn = M / np.clip(M.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(13, 12))
    im = ax.imshow(Mn, cmap="magma_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=4)
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels, fontsize=4)
    ax.set_xlabel("Predicted genus"); ax.set_ylabel("True genus")
    ax.set_title("DINOv2 ViT-L/14 — row-normalised confusion matrix (test)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Fraction of true-genus test images")
    fig.tight_layout()
    save(fig, "fig_dinov2_confusion")


def main():
    print(f"Output → {OUT}")
    fig_macro_ci()
    fig_per_genus()
    fig_training()
    fig_confusion()
    print("Done.")


if __name__ == "__main__":
    main()
