"""
27_reliability_diagram.py — Reliability diagram + calibration direction check (Guo et al., 2017).

Verifies WHETHER the classifier is over- or under-confident, and visualises how temperature
scaling corrects it. The fitted T<1 across all four backbones implies *under*-confidence
(stated confidence systematically LOWER than accuracy — common with a 120-way softmax whose
mass leaks across many classes); this script confirms that on the test set and produces a
paper figure (raw vs temperature-scaled reliability).

Reuses the canonical eval machinery in notebooks/13_evaluate.py.

Run (local DINOv2, the deployed model):
    python notebooks/27_reliability_diagram.py
Other models / Colab: pass --model/--checkpoint/--vocab/--img-size (+ --photos-root).
"""
import argparse, importlib.util, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
# import the digit-prefixed module by path
_spec = importlib.util.spec_from_file_location("ev13", str(Path(__file__).parent / "13_evaluate.py"))
ev = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ev)


def reliability_bins(conf, acc, n_bins=15):
    """Return per-bin (mean_conf, mean_acc, frac) over equal-width confidence bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    mc, ma, fr = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            mc.append(conf[m].mean()); ma.append(acc[m].mean()); fr.append(m.mean())
        else:
            mc.append((lo + hi) / 2); ma.append(np.nan); fr.append(0.0)
    return np.array(mc), np.array(ma), np.array(fr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="timm")
    ap.add_argument("--model", default="vit_large_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--checkpoint",
                    default=str(ROOT / "hf_space/ong-orchid-identifier-v3/models/dinov2l/best_model.pth"))
    ap.add_argument("--vocab",
                    default=str(ROOT / "hf_space/ong-orchid-identifier-v3/models/dinov2l/vocab.json"))
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--splits-dir", default=str(ROOT / "data" / "splits"))
    ap.add_argument("--photos-root", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="use a fixed T; if 0, fit on val_live.csv")
    ap.add_argument("--out", default=str(ROOT / "article" / "fig_reliability.png"))
    ap.add_argument("--label", default="DINOv2 ViT-L/14")
    args = ap.parse_args()

    vocab = json.loads(Path(args.vocab).read_text())
    g2i = {g: i for i, g in enumerate(vocab)}
    model, mean, std = ev.load_model(args.arch, args.model, args.checkpoint, len(vocab), args.img_size)
    tfm = ev.build_tfm(args.img_size, mean, std)

    def logits_for(csv):
        df = pd.read_csv(Path(args.splits_dir) / csv)
        df["path"] = df["path"].apply(lambda p: ev.remap(p, args.photos_root))
        df = df[df["genus"].isin(g2i)].reset_index(drop=True)
        loader = DataLoader(ev.ImgDS(df["path"].tolist(), tfm), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)
        lg = ev.collect_logits(model, loader)
        y = df["genus"].map(lambda g: g2i[g]).to_numpy()
        return lg, y

    # temperature
    if args.temperature > 0:
        T = args.temperature
        print(f"Using fixed T = {T:.4f}")
    else:
        print("Fitting T on val_live.csv ...")
        vlg, vy = logits_for("val_live.csv")
        T = ev.fit_temperature(vlg, vy)
        print(f"  T* = {T:.4f}")

    print("Scoring test_live.csv ...")
    lg, y = logits_for("test_live.csv")
    p_raw = ev.np_softmax(lg)
    p_ts = ev.np_softmax(lg / T)

    def summarise(p, tag):
        conf, pred = p.max(1), p.argmax(1)
        acc = (pred == y).astype(float)
        ece = ev.expected_calibration_error(p, y)
        print(f"  [{tag}] mean_conf={conf.mean():.4f}  accuracy={acc.mean():.4f}  "
              f"gap(conf-acc)={conf.mean()-acc.mean():+.4f}  ECE={ece:.4f}")
        return conf, acc, ece

    print(f"\n=== {args.label} (n={len(y)}) ===")
    c0, a0, e0 = summarise(p_raw, "raw")
    c1, a1, e1 = summarise(p_ts, f"TS (T={T:.3f})")
    direction = "UNDER-confident (conf < accuracy)" if c0.mean() < a0.mean() else "OVER-confident (conf > accuracy)"
    print(f"\nDIRECTION: the model is {direction}.")
    print(f"  -> T={'<' if T < 1 else '>='}1 ({T:.3f}) "
          f"{'sharpens (raises)' if T < 1 else 'softens (lowers)'} confidence to match accuracy.")

    # ── figure ─────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mc0, ma0, _ = reliability_bins(c0, a0)
    mc1, ma1, _ = reliability_bins(c1, a1)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
    ax.plot(mc0, ma0, "o-", color="#c0392b", lw=1.8, ms=4, label=f"raw (ECE {e0:.3f})")
    ax.plot(mc1, ma1, "s-", color="#27ae60", lw=1.8, ms=4, label=f"temperature-scaled, T={T:.2f} (ECE {e1:.3f})")
    ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title(f"Reliability diagram — {args.label}")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(args.out, dpi=200); plt.close(fig)
    print(f"\nSaved figure -> {args.out}")


if __name__ == "__main__":
    main()
