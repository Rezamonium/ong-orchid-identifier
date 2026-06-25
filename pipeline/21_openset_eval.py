"""
21_openset_eval.py — Embedding-space open-set (novel-genus) detection for the deployed DINOv2.

Question: with the deployed model FIXED, can a query whose genus is absent from the reference
bank be flagged by its distance to the nearest known embedding? We answer it comprehensively
with leave-one-genus-out over EVERY genus (not a single cherry-picked one):

  For each genus g (support >= MIN_N):
    - "known" reference = all embeddings whose genus != g
    - score(query) = cosine DISTANCE to nearest known embedding (1 - max cosine sim, self excluded)
    - positives  = images of g (should score HIGH)   |  negatives = images of other genera (LOW)
    - AUROC = separability of the two distance distributions

Reports the distribution of per-genus AUROC (mean/median) — the honest version of the
single-genus "AUROC = 0.87" panel.

IMPORTANT framing caveat: the backbone was fine-tuned on all 120 genera, so this measures the
open-set separability available to a FIXED deployed model at the retrieval layer — useful for
abstain/flag-for-review in production. It is NOT a test of generalisation to genera entirely
unseen during training (that needs leave-K-genera-out RETRAINING; see notebook 22 stub / paper
§3.5). Both are legitimate; this one is free.

Inputs:  retrieval_global/ref_emb.npy (N,1024 L2-normalised) + metadata.json
Outputs (-> _comparison/):
  openset_per_genus_auroc.csv
  figures/fig_openset_auroc_hist.png/.pdf       distribution of per-genus AUROC
  figures/fig_openset_distance_example.png/.pdf  the (b)-style panel for a chosen genus

Run:  python notebooks/21_openset_eval.py
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
from sklearn.metrics import roc_auc_score

ONG = Path(r"E:/Claude Code/ONG_v3")
RETRIEVAL = (ONG / "colab/result/eval/dinov2l_global"
             / "dino retrieval folder_global-20260623T213740Z-3-001" / "retrieval_global")
OUT = ONG / "colab/result/eval/_comparison"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MIN_N = 10        # genera with >= this many images are scored as held-out "unknown"
TOPK = 64         # nearest non-self neighbours cached per image (>= deepest same-genus run)
EXAMPLE_GENUS = "Paphiopedilum"   # panel to mirror the original figure


def load():
    emb = np.load(RETRIEVAL / "ref_emb.npy").astype("float32")
    with open(RETRIEVAL / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    genera = np.array([m["genus"] for m in meta])
    return emb, genera


def topk_neighbours(emb, k):
    """For every row: indices of its k most-similar OTHER rows (self excluded), + similarities.
    Chunked so the full N×N matrix is never materialised."""
    n = len(emb)
    idx_out = np.empty((n, k), dtype="int32")
    sim_out = np.empty((n, k), dtype="float32")
    step = 2000
    for s in range(0, n, step):
        e = min(s + step, n)
        block = emb[s:e] @ emb.T                      # (b, N)
        for r in range(e - s):
            block[r, s + r] = -np.inf                 # exclude self
        part = np.argpartition(-block, k, axis=1)[:, :k]
        rows = np.arange(e - s)[:, None]
        sims = block[rows, part]
        order = np.argsort(-sims, axis=1)             # sort the k by similarity desc
        idx_out[s:e] = part[rows, order]
        sim_out[s:e] = sims[rows, order]
        print(f"  neighbours {e}/{n}", end="\r")
    print()
    return idx_out, sim_out


def main():
    emb, genera = load()
    n = len(emb)
    uniq, counts = np.unique(genera, return_counts=True)
    gid = {g: i for i, g in enumerate(uniq)}
    genus_id = np.array([gid[g] for g in genera])
    print(f"Loaded {n:,} embeddings, {len(uniq)} genera. Caching top-{TOPK} neighbours…")

    nbr_idx, nbr_sim = topk_neighbours(emb, TOPK)
    nbr_genus = genus_id[nbr_idx]                      # (N, k) genus of each neighbour

    eval_genera = [g for g, c in zip(uniq, counts) if c >= MIN_N]
    rows = []
    dist_cache = {}                                   # genus -> (dist_unknown, dist_known) for plotting
    for g in eval_genera:
        c = gid[g]
        # nearest-known similarity per row = first cached neighbour whose genus != c
        known_mask = nbr_genus != c                   # (N, k)
        first = known_mask.argmax(axis=1)             # first True col (0 if none)
        has_known = known_mask.any(axis=1)
        nearest_sim = nbr_sim[np.arange(n), first]
        nearest_sim = np.where(has_known, nearest_sim, -1.0)   # all-g neighbourhood → very far
        dist = 1.0 - nearest_sim
        labels = (genus_id == c).astype(int)
        auroc = roc_auc_score(labels, dist)
        rows.append((g, int((genus_id == c).sum()), round(auroc, 4)))
        if g == EXAMPLE_GENUS:
            dist_cache[g] = (dist[labels == 1], dist[labels == 0])

    rows.sort(key=lambda r: r[2], reverse=True)
    aurocs = np.array([r[2] for r in rows])
    print(f"\nLeave-one-genus-out open-set detection over {len(rows)} genera (n>={MIN_N}):")
    print(f"  mean AUROC   = {aurocs.mean():.3f}")
    print(f"  median AUROC = {np.median(aurocs):.3f}")
    print(f"  range        = {aurocs.min():.3f}–{aurocs.max():.3f}")
    print(f"  fraction >0.8 = {(aurocs > 0.8).mean():.2f}")

    with open(OUT / "openset_per_genus_auroc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["genus", "n", "auroc"])
        w.writerows(rows)

    # ── Figure: distribution of per-genus AUROC ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(aurocs, bins=20, color="#2c6fbb", alpha=0.85, edgecolor="white")
    ax.axvline(aurocs.mean(), color="#d1495b", lw=2,
               label=f"mean = {aurocs.mean():.3f}")
    ax.axvline(0.5, color="#999", ls=":", lw=1.5, label="chance (0.5)")
    ax.set_xlabel("Per-genus open-set AUROC (leave-one-genus-out)")
    ax.set_ylabel("Number of genera")
    ax.set_title(f"DINOv2 embedding-space novel-genus detection\n"
                 f"({len(rows)} genera, n≥{MIN_N})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_openset_auroc_hist.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig_openset_auroc_hist.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote fig_openset_auroc_hist.png")

    # ── Figure: example (b)-style distance histogram ─────────────────────────────
    if EXAMPLE_GENUS in dist_cache:
        du, dk = dist_cache[EXAMPLE_GENUS]
        ex_auroc = next(r[2] for r in rows if r[0] == EXAMPLE_GENUS)
        fig, ax = plt.subplots(figsize=(6, 5))
        bins = np.linspace(0, max(du.max(), dk.max()) * 1.02, 36)
        ax.hist(dk, bins=bins, color="#2c6fbb", alpha=0.7, density=True,
                label=f"in-distribution (known genera, n={len(dk):,})")
        ax.hist(du, bins=bins, color="#d1495b", alpha=0.7, density=True,
                label=f"{EXAMPLE_GENUS} held-out (n={len(du)})")
        ax.set_xlabel("Cosine distance to nearest known embedding")
        ax.set_ylabel("Density")
        ax.set_title(f"Open-set distance to known set\n"
                     f"(leave-one-genus-out: {EXAMPLE_GENUS}; AUROC = {ex_auroc:.2f})")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / "fig_openset_distance_example.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIG / "fig_openset_distance_example.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote fig_openset_distance_example.png ({EXAMPLE_GENUS} AUROC={ex_auroc:.3f})")

    print(f"\nDone -> {OUT}")


if __name__ == "__main__":
    main()
