"""
28_deploy_openset_tau.py — Recalibrate the open-set novelty threshold tau for the DEPLOYED model.

The thresholds in 25_openset_threshold.py (tau=0.29 @5%FPR, 0.21 @10%FPR) were derived on the
STRICT leave-12-genera-out hold-out model (108 known genera). The deployed Gradio app searches a
reference bank of ALL 120 genera, whose nearest-neighbour distance distribution is denser/different,
so tau must be re-fitted on the deployed embeddings the app actually searches.

For each deployed reference vector we compute the cosine distance to its nearest OTHER reference
(leave-one-out = exclude the diagonal), then read tau at the 99/95/90th percentiles = known-genus
false-positive rates of 1/5/10%.

CAVEATS (deployment-specific):
  * No TPR measurable here — the deployed bank has no genuinely-unseen genera, so this script ONLY
    fixes the false-positive operating point. The ~70%@5% / ~94%@10% catch rates carry over
    (approximately) from the strict analysis in 25_openset_threshold.py.
  * Leave-one-out under-estimates the real held-out-known distance (at inference the query is NOT in
    the bank), so the measured FPR is a mild under-estimate → the soft gate fires a touch more often
    than the nominal rate. Acceptable. Expect tau LOWER than the strict model's 0.29 (denser bank).

Run:  python notebooks/28_deploy_openset_tau.py
In:   hf_space/ong-orchid-identifier-v3/models/{ref_emb.npy, ong_metadata.json}
Out:  notebooks/28_deploy_openset_tau.json  (+ printed table; paste fpr_05/fpr_10 into app.py)
"""
import json, sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "hf_space" / "ong-orchid-identifier-v3" / "models"
EMB_PATH = MODELS / "ref_emb.npy"
META_PATH = MODELS / "ong_metadata.json"
OUT_PATH = Path(__file__).resolve().parent / "28_deploy_openset_tau.json"

emb = np.load(EMB_PATH).astype(np.float32)
emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)   # cosine via inner product
meta = json.load(open(META_PATH, encoding="utf-8"))
n_genera = len({m["genus"] for m in meta})
N = len(emb)
assert N == len(meta), f"emb/meta length mismatch: {N} vs {len(meta)}"

# leave-one-out nearest-reference cosine distance over the WHOLE bank (exclude self/diagonal)
dist = np.empty(N, np.float32)
CH = 1000
for s in range(0, N, CH):
    sims = emb[s:s + CH] @ emb.T                      # (chunk, N)
    for li, gi in enumerate(range(s, min(s + CH, N))):
        sims[li, gi] = -np.inf                        # leave-one-out: drop the self-match
    dist[s:s + CH] = 1.0 - sims.max(axis=1)

tau = {q: float(np.percentile(dist, p)) for q, p in (("fpr_01", 99), ("fpr_05", 95), ("fpr_10", 90))}

print(f"N={N}  genera={n_genera}")
print("Deployment tau (percentile of nearest-neighbour distance over all references):")
print(" targetFPR   tau")
print(f"   1%       {tau['fpr_01']:.4f}")
print(f"   5%       {tau['fpr_05']:.4f}   <- Conservative (paste into app.py TAU_05)")
print(f"  10%       {tau['fpr_10']:.4f}   <- Sensitive    (paste into app.py TAU_10)")

OUT_PATH.write_text(json.dumps({
    "n": N, "n_genera": n_genera, "tau": tau,
    "source": "deployed ref_emb.npy (120-genus bank)",
    "note": "FPR operating point only; TPR (~70%@5%, ~94%@10%) carried over from the strict "
            "leave-12-genera-out analysis (25_openset_threshold.py). LOO under-estimates real FPR "
            "slightly (query not in bank at inference)."
}, indent=2))
print(f"\nSaved -> {OUT_PATH}")
