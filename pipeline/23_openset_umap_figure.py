"""
23_openset_umap_figure.py — Combined embedding-structure + open-set separation figure
for the STRICT leave-K-genera-out retrain test (paper §3.5(ii)).

Uses the HOLD-OUT model embeddings (DINOv2 retrained with K genera removed from training):
  colab/openset_logo/openset result/ref_emb_logo.npy   (N, 1024) float32, L2-normalised
  colab/openset_logo/openset result/metadata.json      list of {genus, species}, len N
  colab/openset_logo/held_out_genera.json              the K genera unseen in training

Two-panel figure (mirrors the reference layout):
  (a) UMAP of all N embeddings, top-K *known* genera coloured, rest grey, with the open-set
      exemplar genus (Paphiopedilum, unseen in training) overlaid as red stars.
  (b) Distance-to-nearest-known histogram: in-distribution (known genera) vs the exemplar
      held-out genus, annotated with its open-set AUROC.

Output → colab/result/eval/_comparison/figures/ :
  fig_openset_umap_combined.png / .pdf
  fig_openset_umap_coords.csv   (x, y, genus, dist_to_known, is_heldout)

Run:  python notebooks/23_openset_umap_figure.py
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
from matplotlib.lines import Line2D

ONG = Path(r"E:/Claude Code/ONG_v3")
OS_RES = ONG / "colab/openset_logo/openset result"
HELD_JSON = ONG / "colab/openset_logo/held_out_genera.json"
OUT = ONG / "colab/result/eval/_comparison/figures"
EXEMPLAR = "Paphiopedilum"     # open-set exemplar shown as stars + histogram (matches §3.5 pilot)
TOPK_KNOWN = 12                # number of known genera to colour in panel (a)
NBR = 64                       # neighbours scanned for nearest-known distance
SEED = 42


def load():
    emb = np.load(OS_RES / "ref_emb_logo.npy").astype("float32")
    meta = json.load(open(OS_RES / "metadata.json", encoding="utf-8"))
    genera = np.array([m["genus"] for m in meta])
    held = json.loads(HELD_JSON.read_text())["held_out"]
    assert len(genera) == len(emb), f"emb {len(emb)} != meta {len(genera)}"
    return emb, genera, held


def nearest_known_distance(emb, genera, held):
    """Cosine distance (1 - sim) from each row to its nearest non-held-out (known) neighbour."""
    held_set = set(held)
    is_known_row = ~np.isin(genera, list(held_set))
    n = len(emb)
    dist = np.empty(n, "float32")
    for s in range(0, n, 2000):
        e = min(s + 2000, n)
        block = emb[s:e] @ emb.T            # (chunk, N) cosine sims
        for r in range(e - s):
            block[r, s + r] = -np.inf       # exclude self
        block[:, ~is_known_row] = -np.inf   # only KNOWN neighbours count
        dist[s:e] = 1.0 - block.max(axis=1)
    return dist


def project(emb):
    from sklearn.decomposition import PCA
    import umap
    x50 = PCA(n_components=50, random_state=SEED).fit_transform(emb)
    print("Projecting with UMAP (cosine, PCA-50)…")
    return umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                     random_state=SEED).fit_transform(x50)


def main():
    from sklearn.metrics import roc_auc_score
    OUT.mkdir(parents=True, exist_ok=True)
    emb, genera, held = load()
    held_set = set(held)
    print(f"Loaded {len(emb):,} embeddings · {emb.shape[1]}-d · {len(set(genera))} genera "
          f"· {len(held)} held-out")

    dist = nearest_known_distance(emb, genera, held)
    xy = project(emb)

    # panel (a) colour set: top-K most populous KNOWN (non-held-out) genera
    known_genera = [g for g in genera if g not in held_set]
    uniq, counts = np.unique(known_genera, return_counts=True)
    top = [g for g, _ in sorted(zip(uniq, counts), key=lambda kv: kv[1], reverse=True)[:TOPK_KNOWN]]
    palette = plt.cm.tab20(np.linspace(0, 1, len(top)))

    is_exemplar = genera == EXEMPLAR
    is_known = ~np.isin(genera, list(held_set))
    ex_dist = dist[is_exemplar]
    known_dist = dist[is_known]
    y = np.r_[np.ones(len(ex_dist)), np.zeros(len(known_dist))]
    sc = np.r_[ex_dist, known_dist]
    ex_auroc = roc_auc_score(y, sc)
    print(f"{EXEMPLAR}: n={is_exemplar.sum()}  open-set AUROC vs known = {ex_auroc:.3f}")

    # ── figure ──────────────────────────────────────────────────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.5, 7.2),
                                 gridspec_kw={"width_ratios": [1.35, 1.0]})

    # (a) UMAP
    other = ~np.isin(genera, top) & ~is_exemplar
    a1.scatter(xy[other, 0], xy[other, 1], s=3, c="#dcdcdc", linewidths=0, rasterized=True)
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6,
                      markerfacecolor="#dcdcdc", markeredgecolor="none", label="other genera")]
    for g, col in zip(top, palette):
        m = genera == g
        a1.scatter(xy[m, 0], xy[m, 1], s=7, c=[col], linewidths=0, rasterized=True)
        handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=6,
                              markerfacecolor=col, markeredgecolor="none", label=g))
    a1.scatter(xy[is_exemplar, 0], xy[is_exemplar, 1], s=130, marker="*",
               c="#b30000", edgecolors="k", linewidths=0.5, zorder=5, rasterized=True)
    handles.append(Line2D([0], [0], marker="*", linestyle="", markersize=13,
                          markerfacecolor="#b30000", markeredgecolor="k",
                          label=f"{EXEMPLAR} (held-out / open-set)"))
    a1.set_title(f"(a) UMAP of DINOv2 embeddings (n = {len(emb):,}, {emb.shape[1]}-D),\n"
                 f"coloured by genus", fontsize=12)
    a1.set_xlabel("UMAP-1"); a1.set_ylabel("UMAP-2")
    a1.set_xticks([]); a1.set_yticks([])
    for s in a1.spines.values():
        s.set_visible(False)
    a1.legend(handles=handles, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False, ncol=1)

    # (b) distance histogram
    bins = np.linspace(0, max(ex_dist.max(), known_dist.max()) * 1.02, 40)
    a2.hist(known_dist, bins=bins, color="#2c6fbb", alpha=0.65, density=True,
            label=f"in-distribution\n(known genera, n={len(known_dist):,})")
    a2.hist(ex_dist, bins=bins, color="#d1495b", alpha=0.65, density=True,
            label=f"{EXEMPLAR}\n(held-out, n={len(ex_dist)})")
    a2.set_title(f"(b) Open-set distance to known set\n"
                 f"(leave-{len(held)}-genera-out; {EXEMPLAR} AUROC = {ex_auroc:.2f})", fontsize=12)
    a2.set_xlabel("Cosine distance to nearest known embedding")
    a2.set_ylabel("Density")
    a2.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)

    fig.suptitle("Embedding-space structure and open-set separation in the birdsheadorchid.id "
                 "classifier (DINOv2)", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "fig_openset_umap_combined.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig_openset_umap_combined.pdf", bbox_inches="tight")
    plt.close(fig)

    with open(OUT / "fig_openset_umap_coords.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "genus", "dist_to_known", "is_heldout"])
        for (x, yy), g, d in zip(xy, genera, dist):
            w.writerow([f"{x:.4f}", f"{yy:.4f}", g, f"{d:.4f}", int(g in held_set)])

    print(f"Wrote → {OUT}/fig_openset_umap_combined.png (+ .pdf, coords.csv)")


if __name__ == "__main__":
    main()
