# ============================================================================
# TAHAP A.3 — Evaluasi backbone bake-off (protokol identik)
# NOTE (2026-06-22): lebih disarankan pakai run_bakeoff_L4.ipynb (Sel 3.6 pilot + Sel 5
# summary), yang sudah sinkron. Cell ini dipertahankan sebagai alternatif standalone dan
# kini juga mengevaluasi KEDUA checkpoint (best_model_macro.pth & best_model_global.pth).
# Tiap checkpoint -> eval/<model>[_global]/results.json (+ per_genus, confusion, confused_pairs)
# Retrieval pakai FAISS bila tersedia; jika tidak, fallback numpy (hasil identik).
# ============================================================================
ROOT = "/content/drive/MyDrive/orchid_project"

# faiss opsional untuk Tahap A (numpy fallback exact). Pasang biar lebih cepat:
get_ipython().system('pip -q install faiss-cpu')

# img-size HARUS sama dengan saat training tiap model.
MODELS = [
    # (label,        arch,       timm/openclip model id,                          img,  dir)
    ("effnetv2l",   "timm",     "tf_efficientnetv2_l.in21k_ft_in1k",             448, "effnetv2l"),
    ("convnextv2l", "timm",     "convnextv2_large.fcmae_ft_in22k_in1k",          384, "convnextv2l"),
    ("dinov2l",     "timm",     "vit_large_patch14_reg4_dinov2.lvd142m",         448, "dinov2l"),
    ("bioclip2",    "openclip", "hf-hub:imageomics/bioclip-2",                   224, "bioclip2"),
]

import os
for label, arch, model_id, img, mdir in MODELS:
    vocab = f"{ROOT}/models/{mdir}/vocab.json"
    arch_flag = "--arch openclip " if arch == "openclip" else ""
    # Evaluate BOTH selection checkpoints; fall back to legacy single best_model.pth.
    variants = [(t, f"{ROOT}/models/{mdir}/best_model_{t}.pth") for t in ("macro", "global")
                if os.path.exists(f"{ROOT}/models/{mdir}/best_model_{t}.pth")]
    if not variants:
        variants = [("macro", f"{ROOT}/models/{mdir}/best_model.pth")]
    for tag, ckpt in variants:
        out = f"{ROOT}/eval/{label}" if tag == "macro" else f"{ROOT}/eval/{label}_{tag}"
        print(f"\n{'='*70}\n[EVAL] {label} [{tag}]  ({model_id} @ {img})\n{'='*70}")
        get_ipython().run_line_magic(
            "run",
            f"{ROOT}/scripts/13_evaluate.py {arch_flag}--model {model_id} --img-size {img} "
            f"--checkpoint {ckpt} --vocab {vocab} --photos-root /content/photos "
            f"--out-dir {out}"
        )

# ── Ringkas hasil keempat model jadi satu tabel ─────────────────────────────
import json
from pathlib import Path
print(f"\n{'='*70}\nRINGKASAN — bandingkan vs baseline effb4: macro 74.0% (>=5: 76.2%), spR@5 73.4%\n{'='*70}")
hdr = f"{'model':<14}{'macro@1':>9}{'macro@5':>9}{'spR@5':>8}{'gnR@5':>8}"
print(hdr); print("-"*len(hdr))
labels = []
for label, *_ in MODELS:
    labels.append(label)
    if os.path.isdir(f"{ROOT}/eval/{label}_global"):
        labels.append(f"{label}_global")
for label in labels:
    rj = Path(f"{ROOT}/eval/{label}/results.json")
    if not rj.exists():
        print(f"{label:<14}  (results.json tidak ada)"); continue
    r = json.loads(rj.read_text())
    c = r.get("classification", r); rt = r.get("retrieval") or {}
    def g(d, *keys):
        for k in keys:
            if k in d: return d[k]
        return None
    m1 = g(c, "macro_top1", "macro_acc", "macro_top1_acc")
    m5 = g(c, "macro_top5", "macro_top5_acc")
    sp = rt.get("species_recall@5"); gn = rt.get("genus_recall@5")
    f = lambda x: f"{x*100:>7.1f}%" if isinstance(x,(int,float)) else f"{'--':>8}"
    print(f"{label:<14}{f(m1)[:9]:>9}{f(m5)[:9]:>9}{f(sp)[:8]:>8}{f(gn)[:8]:>8}")
print("\nSelesai. Catat angka ke ONGv3_progress.Rmd, pilih pemenang genus (macro@1) & retrieval (spR@5).")
