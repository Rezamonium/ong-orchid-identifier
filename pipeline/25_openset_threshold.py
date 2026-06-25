"""
25_openset_threshold.py — Pick a deployment threshold for the open-set novelty score.

Uses the strict leave-12-genera-out hold-out model embeddings. For every photo we score
the cosine distance to the nearest KNOWN-genus reference embedding (leave-one-out for the
known photos themselves). Known = the 108 trained genera; unknown = the 12 held-out genera.
A threshold tau is fixed on the known-distance distribution (target false-positive rate),
and we report the resulting true-positive rate on the genuinely-unseen genera.

Run:  python notebooks/25_openset_threshold.py
In:   colab/openset_logo/openset result/{ref_emb_logo.npy, metadata.json}
      colab/openset_logo/held_out_genera.json
"""
import sys, json
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np

BASE = Path(r"E:/Claude Code/ONG_v3/colab/openset_logo")
RES = BASE / "openset result"

emb = np.load(RES / "ref_emb_logo.npy").astype(np.float32)
meta = json.load(open(RES / "metadata.json", encoding="utf-8"))
held = set(json.load(open(BASE / "held_out_genera.json", encoding="utf-8"))["held_out"])
genera = np.array([m["genus"] for m in meta])
unknown = np.array([g in held for g in genera]); known = ~unknown

emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)   # cosine via inner product
known_idx = np.where(known)[0]
K = emb[known_idx]

dist = np.empty(len(emb), np.float32)
CH = 1000
for s in range(0, len(emb), CH):
    sims = emb[s:s + CH] @ K.T
    for li, gi in enumerate(range(s, min(s + CH, len(emb)))):
        pos = np.searchsorted(known_idx, gi)
        if pos < len(known_idx) and known_idx[pos] == gi:
            sims[li, pos] = -np.inf            # leave-one-out for known references
    dist[s:s + CH] = 1.0 - sims.max(axis=1)

kd, ud = dist[known], dist[unknown]
print(f"known={known.sum()}  unknown={unknown.sum()}")
print("Operating points (tau on known-distance percentile):")
print(" targetFPR   tau      FPR     TPR   caught/total")
for q in (99, 95, 90):
    tau = np.percentile(kd, q)
    print("   %2d%%      %.4f   %.3f   %.3f   %d/%d"
          % (100 - q, tau, (kd > tau).mean(), (ud > tau).mean(), int((ud > tau).sum()), len(ud)))
