"""
18_embedding_figure.py — Embedding-space structure of the deployed backbone (DINOv2 ViT-L/14).

Projects the 1024-d, L2-normalised DINOv2 reference embeddings to 2-D with UMAP
(cosine metric, PCA-50 pre-reduction) and plots them coloured by genus. This is the
"does the embedding space have genus structure?" figure that justifies FAISS retrieval.

Inputs (from the deployed retrieval bundle):
  retrieval_global/ref_emb.npy      (N, 1024) float32, L2-normalised
  retrieval_global/metadata.json    list of {species, genus, path}, len N

Outputs → <out-dir> (default: colab/result/eval/_comparison/figures):
  embedding_umap_topgenera.png/.pdf   top-K genera coloured, rest light grey
  embedding_umap_magnets.png          same layout, Dendrobium/Bulbophyllum highlighted
  embedding_umap_coords.csv           x,y,genus,species per point (re-plot without recompute)

Run:
  python notebooks/18_embedding_figure.py
  python notebooks/18_embedding_figure.py --top-k 15 --seed 42
"""

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 chokes on → in prints
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ONG = Path(r"E:/Claude Code/ONG_v3")
RETRIEVAL = (ONG / "colab/result/eval/dinov2l_global"
             / "dino retrieval folder_global-20260623T213740Z-3-001" / "retrieval_global")
MAGNETS = ["Dendrobium", "Bulbophyllum"]  # the two error-sink genera from the confusion analysis


def load(retrieval: Path):
    emb = np.load(retrieval / "ref_emb.npy").astype("float32")
    with open(retrieval / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    genera = np.array([m["genus"] for m in meta])
    species = np.array([m.get("species", "") for m in meta])
    assert len(genera) == len(emb), f"emb {len(emb)} != meta {len(genera)}"
    return emb, genera, species


def project(emb: np.ndarray, seed: int) -> np.ndarray:
    """PCA-50 → UMAP-2D (cosine). Falls back to t-SNE then PCA if UMAP is unavailable."""
    from sklearn.decomposition import PCA
    x50 = PCA(n_components=50, random_state=seed).fit_transform(emb)
    try:
        import umap
        print("Projecting with UMAP (cosine, PCA-50)…")
        return umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                         random_state=seed).fit_transform(x50)
    except Exception as e:
        print(f"UMAP unavailable ({e}); falling back to t-SNE.")
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, init="pca", perplexity=40,
                    random_state=seed).fit_transform(x50)


def scatter(xy, genera, focus, palette, title, path, legend_title):
    """Plot points: genera in `focus` get palette colours, the rest light grey beneath."""
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    other = ~np.isin(genera, focus)
    ax.scatter(xy[other, 0], xy[other, 1], s=3, c="#dcdcdc", linewidths=0, rasterized=True)
    handles = []
    for g, col in zip(focus, palette):
        m = genera == g
        ax.scatter(xy[m, 0], xy[m, 1], s=6, c=[col], linewidths=0, rasterized=True)
        handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=6,
                              markerfacecolor=col, markeredgecolor="none", label=g))
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=handles, title=legend_title, fontsize=7, title_fontsize=8,
              loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    if str(path).endswith(".png"):
        fig.savefig(str(path)[:-4] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", type=Path, default=RETRIEVAL)
    ap.add_argument("--out-dir", type=Path, default=ONG / "colab/result/eval/_comparison/figures")
    ap.add_argument("--top-k", type=int, default=12, help="genera to colour (by photo count)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    emb, genera, species = load(args.retrieval)
    print(f"Loaded {len(emb):,} embeddings · {emb.shape[1]}-d · {len(set(genera))} genera")

    xy = project(emb, args.seed)

    # save coords for reproducible re-plotting
    with open(args.out_dir / "embedding_umap_coords.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "genus", "species"])
        for (x, y), g, sp in zip(xy, genera, species):
            w.writerow([f"{x:.4f}", f"{y:.4f}", g, sp])

    # top-K genera by photo count
    uniq, counts = np.unique(genera, return_counts=True)
    top = [g for g, _ in sorted(zip(uniq, counts), key=lambda kv: kv[1], reverse=True)[:args.top_k]]
    palette = plt.cm.tab20(np.linspace(0, 1, len(top)))

    scatter(xy, genera, top, palette,
            f"DINOv2 ViT-L/14 embedding space (UMAP, N={len(emb):,})",
            args.out_dir / "embedding_umap_topgenera.png",
            f"Top {args.top_k} genera\n(by photo count)")

    # magnet-class highlight (the error sinks)
    scatter(xy, genera, MAGNETS, plt.cm.Set1(np.linspace(0, 0.25, len(MAGNETS))),
            "Magnet classes in DINOv2 embedding space",
            args.out_dir / "embedding_umap_magnets.png",
            "Error-sink genera")

    print(f"Done → {args.out_dir}")


if __name__ == "__main__":
    main()
