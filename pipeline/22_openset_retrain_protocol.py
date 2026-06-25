"""
22_openset_retrain_protocol.py — Leave-K-genera-out open-set GENERALISATION test (the strict one).

Notebook 21 measured open-set separability with the FIXED deployed model (optimistic, because it
trained on all genera). This protocol instead withholds K genera from TRAINING entirely, retrains
DINOv2 on the rest, and scores the withheld genera as genuinely-unseen unknowns — the
deployment-realistic estimate referenced in paper §3.5(ii) and §2.7.8.

Two local steps bracket one Colab retrain:

  python notebooks/22_openset_retrain_protocol.py make-splits
      → picks K held-out genera (stratified by support, seed 42, forces Paphiopedilum),
        writes modified train/val splits with those genera removed, held_out_genera.json,
        and colab/openset_logo/RUN_ON_COLAB.md (the retrain + embedding recipe).

  # ... run the retrain + full-dataset embedding on Colab per RUN_ON_COLAB.md ...

  python notebooks/22_openset_retrain_protocol.py evaluate \
      --emb ref_emb_logo.npy --meta metadata.json
      → open-set AUROC over the held-out (unseen) genera + figure.

Stdlib + numpy + sklearn + matplotlib only.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

ONG = Path(r"E:/Claude Code/ONG_v3")
SPLITS = ONG / "data/splits"
OUTDIR = ONG / "colab/openset_logo"
K = 12                     # number of held-out genera
MIN_TOTAL = 15             # a genus needs this many images total to be an evaluable unknown
MAX_TOTAL = 300            # cap: don't withhold dominant genera (unrealistic as "novel"; would
                           # gut training). Realistic unknowns are rare/under-represented genera.
SEED = 42
FORCE = ["Paphiopedilum"]  # always hold out (connects to the §3.5 pilot figure)
TOPK = 64


# ── make-splits ───────────────────────────────────────────────────────────────────
def read_split(name):
    rows = []
    with open(SPLITS / name, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def make_splits():
    train = read_split("train_live.csv")
    val = read_split("val_live.csv")
    test = read_split("test_live.csv")
    total = {}
    for r in train + val + test:
        total[r["genus"]] = total.get(r["genus"], 0) + 1

    candidates = sorted([g for g, n in total.items() if MIN_TOTAL <= n <= MAX_TOTAL],
                        key=lambda g: total[g])
    # stratify candidates into 3 support tertiles, sample evenly
    rng = np.random.default_rng(SEED)
    thirds = np.array_split(candidates, 3)            # low / mid / high support
    per = max(1, (K - len(FORCE)) // 3)
    picked = set(g for g in FORCE if g in total)
    for band in thirds:
        pool = [g for g in band if g not in picked]
        take = min(per, len(pool))
        picked.update(rng.choice(pool, size=take, replace=False).tolist())
    # top up to K from remaining candidates if rounding left us short
    if len(picked) < K:
        rest = [g for g in candidates if g not in picked]
        picked.update(rng.choice(rest, size=min(K - len(picked), len(rest)),
                                 replace=False).tolist())
    held = sorted(picked, key=lambda g: total[g])

    out_splits = OUTDIR / "data" / "splits"
    out_splits.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train_live.csv", train), ("val_live.csv", val)]:
        kept = [r for r in rows if r["genus"] not in picked]
        with open(out_splits / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(kept)
        print(f"  {name}: {len(rows):,} → {len(kept):,} rows ({len(picked)} genera removed)")

    n_after = len({r['genus'] for r in train if r['genus'] not in picked})
    info = {"held_out": held, "k": len(held), "seed": SEED,
            "support": {g: total[g] for g in held},
            "n_train_genera_after": n_after}
    (OUTDIR / "held_out_genera.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    write_runbook(held, n_after)
    print(f"\nHeld-out genera ({len(held)}): " + ", ".join(f"{g}({total[g]})" for g in held))
    print(f"Training genera after removal: {n_after}")
    print(f"Wrote → {OUTDIR}  (splits, held_out_genera.json, RUN_ON_COLAB.md)")


def write_runbook(held, n_after):
    md = f"""# Leave-{len(held)}-genera-out open-set retrain — Colab runbook

Held-out (UNSEEN in training) genera: {', '.join(held)}
Training genera after removal: {n_after}

## 1. Upload the modified splits to a SEPARATE Drive run-root
Copy `colab/openset_logo/data/splits/{{train_live,val_live}}.csv` to, e.g.
`MyDrive/orchid_openset_logo/data/splits/`. (Keep the original photos/ unchanged — the held-out
genera's image FILES must still exist; they are only removed from the train/val CSVs.)

## 2. Retrain DINOv2 with the held-out genera removed (identical protocol to the paper)
```bash
%run /content/drive/MyDrive/orchid_project/scripts/03_train_bakeoff_colab.py \\
    --model dinov2l \\
    --drive-root /content/drive/MyDrive/orchid_openset_logo \\
    --photos-root /content/photos \\
    --loss ce --sampler-power 0 --select global \\
    --warmup-epochs 3 --finetune-epochs 22 --seed 42
# → /content/drive/MyDrive/orchid_openset_logo/models/dinov2l/best_model_global.pth (+ vocab.json)
```

## 3. Embed the FULL image set with the held-out model (held-out genera included as queries)
```python
import json, numpy as np, torch, timm, pandas as pd
from torchvision import transforms
from PIL import Image
from pathlib import Path

ROOT = Path('/content/drive/MyDrive/orchid_openset_logo')
vocab = json.load(open(ROOT/'models/dinov2l/vocab.json'))      # the (≈{n_after}) trained genera
m = timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m', pretrained=False,
                      num_classes=len(vocab), img_size=448)
sd = torch.load(ROOT/'models/dinov2l/best_model_global.pth', map_location='cpu', weights_only=False)
m.load_state_dict(sd.get('state_dict', sd) if isinstance(sd, dict) else sd, strict=False)
cfg = timm.data.resolve_model_data_config(m); m = m.cuda().eval()
tfm = transforms.Compose([transforms.Resize(int(448*1.1)), transforms.CenterCrop(448),
                          transforms.ToTensor(), transforms.Normalize(cfg['mean'], cfg['std'])])

# embed EVERY live image (train+val+test of ALL 120 genera, including the held-out ones)
df = pd.concat([pd.read_csv('/content/drive/MyDrive/orchid_project/data/splits/%s_live.csv' % s)
                for s in ('train','val','test')], ignore_index=True)
df['path'] = df['path'].str.replace('\\\\','/').apply(
    lambda p: '/content/photos/' + '/'.join(p.split('/')[-2:]))
embs, meta = [], []
with torch.no_grad():
    for i in range(0, len(df), 64):
        batch = df.iloc[i:i+64]
        x = torch.stack([tfm(Image.open(p).convert('RGB')) for p in batch['path']]).cuda()
        e = m.forward_head(m.forward_features(x), pre_logits=True)
        e = torch.nn.functional.normalize(e.float(), dim=1).cpu().numpy()
        embs.append(e); meta += batch[['genus','species','path']].to_dict('records')
np.save(ROOT/'ref_emb_logo.npy', np.concatenate(embs).astype('float32'))
json.dump(meta, open(ROOT/'metadata.json','w'))
print('saved', ROOT/'ref_emb_logo.npy', len(meta))
```

## 4. Download `ref_emb_logo.npy` + `metadata.json`, then locally:
```bash
python notebooks/22_openset_retrain_protocol.py evaluate \\
    --emb <path>/ref_emb_logo.npy --meta <path>/metadata.json
```
This scores the held-out genera (now genuinely unseen) and writes the strict AUROC + figure.
"""
    (OUTDIR / "RUN_ON_COLAB.md").write_text(md, encoding="utf-8")


# ── evaluate ──────────────────────────────────────────────────────────────────────
def topk_neighbours(emb, k):
    n = len(emb)
    idx = np.empty((n, k), "int32"); sim = np.empty((n, k), "float32")
    for s in range(0, n, 2000):
        e = min(s + 2000, n)
        block = emb[s:e] @ emb.T
        for r in range(e - s):
            block[r, s + r] = -np.inf
        part = np.argpartition(-block, k, axis=1)[:, :k]
        rows = np.arange(e - s)[:, None]
        order = np.argsort(-block[rows, part], axis=1)
        idx[s:e] = part[rows, order]; sim[s:e] = block[rows, part][rows, order]
    return idx, sim


def evaluate(emb_path, meta_path):
    from sklearn.metrics import roc_auc_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    held = json.loads((OUTDIR / "held_out_genera.json").read_text())["held_out"]
    held_set = set(held)
    emb = np.load(emb_path).astype("float32")
    meta = json.load(open(meta_path, encoding="utf-8"))
    genera = np.array([m["genus"] for m in meta])
    assert len(genera) == len(emb), f"emb {len(emb)} vs meta {len(genera)}"
    print(f"Loaded {len(emb):,} embeddings; held-out genera: {', '.join(held)}")

    uniq = sorted(set(genera)); gid = {g: i for i, g in enumerate(uniq)}
    gi = np.array([gid[g] for g in genera])
    held_ids = {gid[g] for g in held_set if g in gid}

    nbr_idx, nbr_sim = topk_neighbours(emb, TOPK)
    nbr_g = gi[nbr_idx]
    # nearest KNOWN (= non-held-out) neighbour per row
    known_mask = ~np.isin(nbr_g, list(held_ids))
    first = known_mask.argmax(axis=1)
    has_known = known_mask.any(axis=1)
    nearest = np.where(has_known, nbr_sim[np.arange(len(emb)), first], -1.0)
    dist = 1.0 - nearest

    is_unknown = np.isin(gi, list(held_ids))
    known_dist = dist[~is_unknown]

    rows = []
    for g in held:
        if g not in gid:
            continue
        gd = dist[gi == gid[g]]
        y = np.r_[np.ones(len(gd)), np.zeros(len(known_dist))]
        s = np.r_[gd, known_dist]
        rows.append((g, len(gd), round(roc_auc_score(y, s), 4)))
    pooled = roc_auc_score(is_unknown.astype(int), dist)
    aur = np.array([r[2] for r in rows])
    print(f"\nLeave-{len(rows)}-genera-out (UNSEEN) open-set detection:")
    print(f"  per-genus mean AUROC = {aur.mean():.3f}  median = {np.median(aur):.3f}")
    print(f"  range = {aur.min():.3f}–{aur.max():.3f}   pooled AUROC = {pooled:.3f}")

    OUTDIR.mkdir(exist_ok=True)
    with open(OUTDIR / "openset_logo_auroc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["genus", "n", "auroc"]); w.writerows(rows)

    # 2-panel: (a) per-genus AUROC, (b) example distance histogram
    ex = "Paphiopedilum" if any(r[0] == "Paphiopedilum" for r in rows) else \
        min(rows, key=lambda r: abs(r[2] - np.median(aur)))[0]
    ex_auroc = next(r[2] for r in rows if r[0] == ex)
    du = dist[gi == gid[ex]]
    order = sorted(rows, key=lambda r: r[2])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.barh([r[0] for r in order], [r[2] for r in order], color="#2c6fbb")
    a1.axvline(aur.mean(), color="#d1495b", lw=2, label=f"mean = {aur.mean():.3f}")
    a1.axvline(0.5, color="#999", ls=":", lw=1.5, label="chance")
    a1.set_xlim(0, 1); a1.set_xlabel("Open-set AUROC (genus unseen in training)")
    a1.set_title(f"(a) Per-genus AUROC  (pooled = {pooled:.3f})"); a1.legend(frameon=False)

    bins = np.linspace(0, max(du.max(), known_dist.max()) * 1.02, 36)
    a2.hist(known_dist, bins=bins, color="#2c6fbb", alpha=0.7, density=True,
            label=f"in-distribution (known, n={len(known_dist):,})")
    a2.hist(du, bins=bins, color="#d1495b", alpha=0.7, density=True,
            label=f"{ex} held-out (n={len(du)})")
    a2.set_xlabel("Cosine distance to nearest known embedding"); a2.set_ylabel("Density")
    a2.set_title(f"(b) Distance to known set: {ex}  (AUROC = {ex_auroc:.2f})")
    a2.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Leave-{len(rows)}-genera-out open-set detection (DINOv2; genera unseen in "
                 f"training)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_openset_logo.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTDIR / "fig_openset_logo.pdf", bbox_inches="tight")
    print(f"Wrote → {OUTDIR}/openset_logo_auroc.csv, fig_openset_logo.png")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make-splits")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--emb", required=True)
    ev.add_argument("--meta", required=True)
    args = ap.parse_args()
    if args.cmd == "make-splits":
        make_splits()
    else:
        evaluate(args.emb, args.meta)


if __name__ == "__main__":
    main()
