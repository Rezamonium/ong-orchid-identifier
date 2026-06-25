"""
Model Evaluation — ONG Orchid Identifier v3
============================================
Honest, backbone-agnostic evaluation so the Phase-2 backbone bake-off
(BioCLIP-2 / DINOv2 / ConvNeXt-V2 / EfficientNetV2) can be compared on an
identical protocol. Run once per trained model.

Two independent evaluations:

1. GENUS CLASSIFICATION  (on test_live.csv — species-stratified, no leakage)
   - Global  top-1 / top-5
   - MACRO   top-1 / top-5   (mean over genera — exposes failure on rare genera
                              that global accuracy hides behind Bulbo/Dendro)
   - Balanced accuracy
   - Per-genus accuracy + support           → per_genus_accuracy.csv + .png
   - Confusion matrix (normalised)          → confusion_matrix.csv + .png
   - Top-20 most-confused genus pairs       → confused_pairs.csv
   - Expected Calibration Error (ECE)       (is the shown confidence honest?)

2. SPECIES RETRIEVAL  (FAISS embedding quality)
   NOTE: cannot use the species-stratified split (a test species has 0 photos
   in train → recall always 0). Instead we build a *photo-level* holdout from
   all_images.csv: for every species with >=2 photos, 1 photo (seeded) is held
   out as a QUERY, the rest form the REFERENCE database. This measures the real
   deployment question: "given a new photo of a known species, do we retrieve
   the right species?"
   - Recall@1 / @5 / @10  (species level, micro + macro)
   - Recall@1 / @5        (genus level)

Embeddings use the model's pre-logits features (timm forward_head pre_logits),
so the SAME backbone that classifies also produces the retrieval embedding.
(After Phase 3, point --embed-checkpoint at the ArcFace head to evaluate that.)

Usage (baseline EfficientNet-B4):
    python notebooks/13_evaluate.py

Bake-off example:
    python notebooks/13_evaluate.py --model convnextv2_large.fcmae_ft_in22k_in1k \\
        --checkpoint models/convnextv2_large/best_model.pth \\
        --vocab models/convnextv2_large/vocab.json --img-size 448

BioCLIP-2 (open_clip) example:
    python notebooks/13_evaluate.py --arch openclip \\
        --model hf-hub:imageomics/bioclip-2 --img-size 224 \\
        --checkpoint models/bioclip2/best_model.pth --vocab models/bioclip2/vocab.json

On Colab, add  --photos-root /content/photos  to remap the CSV paths.
"""

import argparse, json, random, sys
from pathlib import Path
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")   # allow unicode prints on Windows cp1252
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True   # tolerate slightly truncated JPEGs

# ── Defaults (baseline model) ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEF_DIR      = PROJECT_ROOT / "hf_space" / "models" / "efficientnet_b4"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLIP_MEAN, CLIP_STD = ([0.48145466, 0.4578275, 0.40821073],
                       [0.26862954, 0.26130258, 0.27577711])


def get_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--arch", choices=["timm", "openclip"], default="timm",
                    help="model family; 'openclip' for BioCLIP-2 trained by 03_train_bakeoff")
    ap.add_argument("--model",      default="efficientnet_b4",
                    help="timm model name, or open_clip name e.g. hf-hub:imageomics/bioclip-2")
    ap.add_argument("--checkpoint", default=str(DEF_DIR / "best_model.pth"))
    ap.add_argument("--vocab",      default=str(DEF_DIR / "vocab.json"))
    ap.add_argument("--img-size",   type=int, default=380)
    ap.add_argument("--splits-dir", default=str(PROJECT_ROOT / "data" / "splits"))
    ap.add_argument("--out-dir",    default=None, help="default: eval/<model>")
    ap.add_argument("--photos-root", default=None,
                    help="Replace '<...>/photos' prefix in CSV paths (e.g. /content/photos on Colab)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers",    type=int, default=2)
    ap.add_argument("--no-retrieval", action="store_true", help="skip species retrieval eval")
    ap.add_argument("--min-photos", type=int, default=2,
                    help="species need >= this many photos to contribute a query")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap #images for a quick smoke test (0 = full eval)")
    ap.add_argument("--synonyms", default=str(PROJECT_ROOT / "data" / "taxonomy_synonyms.csv"),
                    help="taxonomy_synonyms.csv; confirmed rows pool synonymous output classes "
                         "so an unmerged-vocab baseline stays comparable to merged-label models")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="bootstrap resamples for 95%% CIs on the metrics (0 = off)")
    ap.add_argument("--fit-temperature", action="store_true",
                    help="fit a single post-hoc temperature on the val split (Guo et al., 2017) "
                         "and report test ECE before/after; top-1/accuracy are unchanged")
    ap.add_argument("--temp-split", default="val_live.csv",
                    help="split CSV used to fit the temperature (default: val_live.csv)")
    ap.add_argument("--seed",       type=int, default=42)
    return ap.parse_args()


def load_synonyms(path):
    """Return {from_genus: to_genus} for rows with status == 'confirmed'."""
    import csv
    p = Path(path)
    if not p.exists():
        return {}
    syn = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status", "").strip().lower() == "confirmed" and r.get("to_genus", "").strip():
                syn[r["from_genus"].strip()] = r["to_genus"].strip()
    return syn


# ── Data ──────────────────────────────────────────────────────────────────────
def remap(path: str, photos_root: str | None) -> str:
    if not photos_root:
        return path
    parts = path.replace("\\", "/").split("/")
    low = [p.lower() for p in parts]
    if "photos" in low:
        i = low.index("photos")
        return photos_root.rstrip("/") + "/" + "/".join(parts[i + 1:])
    return path


class ImgDS(Dataset):
    def __init__(self, paths, tfm):
        self.paths, self.tfm = paths, tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tfm(img), i


def build_tfm(img_size: int, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ── Model ─────────────────────────────────────────────────────────────────────
class BioClipClassifier(torch.nn.Module):
    """Mirror of the wrapper in 03_train_bakeoff_colab.py so its checkpoints load."""
    def __init__(self, open_clip_model, embed_dim, n_cls):
        super().__init__()
        self.visual = open_clip_model.visual
        self.head = torch.nn.Linear(embed_dim, n_cls)
    def embed(self, x):
        return self.visual(x)
    def forward(self, x):
        return self.head(self.embed(x))


def load_model(arch, name, checkpoint, n_cls, img_size):
    """Returns (model, mean, std). timm → norm from model config; openclip → CLIP norm.

    NOTE: img_size is passed to timm for ViT models so the position-embedding grid
    matches the training checkpoint (e.g. DINOv2 trained @448 ≠ default 518), and
    mean/std are resolved from the model's own config (e.g. tf_efficientnetv2_l uses
    Inception 0.5/0.5/0.5, not ImageNet) — both must mirror 03_train_bakeoff_colab.py.
    """
    if arch == "openclip":
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms(name)
        model = BioClipClassifier(clip_model, clip_model.visual.output_dim, n_cls)
        mean, std = CLIP_MEAN, CLIP_STD
    else:
        import timm
        kwargs = dict(pretrained=False, num_classes=n_cls)
        if "vit" in name:
            kwargs["img_size"] = img_size      # match interpolated pos-embed from training
        model = timm.create_model(name, **kwargs)
        cfg = timm.data.resolve_model_data_config(model)
        mean, std = list(cfg["mean"]), list(cfg["std"])
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(DEVICE).eval(), mean, std


def _emb_and_logits(model, imgs, want_logits):
    """Pre-logits embedding (+ optional logits) for either model family."""
    if isinstance(model, BioClipClassifier):
        emb = model.embed(imgs)
        logits = model.head(emb) if want_logits else None
    else:
        feat = model.forward_features(imgs)
        emb  = model.forward_head(feat, pre_logits=True)
        logits = model.get_classifier()(emb) if want_logits else None
    return emb, logits


@torch.no_grad()
def forward_all(model, loader, want_logits=True, want_emb=False):
    """Single pass → (probs[N,C] or None, embeddings[N,D] or None) in dataset order."""
    probs_out, emb_out, order = [], [], []
    use_amp = DEVICE.type == "cuda"
    for imgs, idx in tqdm(loader, leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.autocast(DEVICE.type, enabled=use_amp):
            emb, logits = _emb_and_logits(model, imgs, want_logits)
            if want_logits:
                probs_out.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        if want_emb:
            emb_out.append(emb.float().cpu().numpy())
        order.append(idx.numpy())
    order = np.concatenate(order)
    inv = np.argsort(order)
    probs = np.concatenate(probs_out)[inv] if want_logits else None
    embs  = np.concatenate(emb_out)[inv]   if want_emb    else None
    return probs, embs


@torch.no_grad()
def collect_logits(model, loader):
    """Single pass → raw (pre-softmax) logits [N, C] in dataset order (for temperature fitting)."""
    out, order = [], []
    use_amp = DEVICE.type == "cuda"
    for imgs, idx in tqdm(loader, leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.autocast(DEVICE.type, enabled=use_amp):
            _, logits = _emb_and_logits(model, imgs, want_logits=True)
        out.append(logits.float().cpu().numpy())
        order.append(idx.numpy())
    order = np.concatenate(order)
    return np.concatenate(out)[np.argsort(order)]


def np_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits, labels, max_iter=200):
    """Single scalar T>0 minimising NLL of softmax(logits/T) on a held-out split
    (Guo et al., 2017). Parameterised T=exp(s) so LBFGS is unconstrained and stable."""
    lg = torch.tensor(logits, dtype=torch.float32)
    lb = torch.tensor(labels, dtype=torch.long)
    s = torch.zeros(1, requires_grad=True)            # T = exp(s); start at T = 1
    opt = torch.optim.LBFGS([s], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    nll = torch.nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad()
        loss = nll(lg / s.exp(), lb)
        loss.backward()
        return loss
    opt.step(closure)
    return float(s.exp().item())


# ── Metrics helpers ───────────────────────────────────────────────────────────
def expected_calibration_error(probs, labels, n_bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc  = (pred == labels).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            ece += (m.mean()) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def _pct_ci(samples):
    a = np.asarray(samples)
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def bootstrap_classification(labels, correct1, correct5, n_boot, seed=42):
    """Image-level bootstrap → 95% CIs for global/macro top-1 and macro top-5.
    Macro is averaged over genera present in each resample (rare genera → wider CI)."""
    rng = np.random.default_rng(seed)
    N = len(labels)
    g1, m1, m5 = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        l, c1, c5 = labels[idx], correct1[idx], correct5[idx]
        g1.append(c1.mean())
        cls = np.unique(l)
        m1.append(np.mean([c1[l == c].mean() for c in cls]))
        m5.append(np.mean([c5[l == c].mean() for c in cls]))
    return {"global_top1": _pct_ci(g1), "macro_top1": _pct_ci(m1), "macro_top5": _pct_ci(m5)}


def bootstrap_mean(flags, n_boot, seed=42):
    """Bootstrap 95% CI for a simple mean of boolean flags (e.g. a recall@k)."""
    a = np.asarray(flags, dtype=float)
    N = len(a)
    if N == 0 or n_boot == 0:
        return None
    rng = np.random.default_rng(seed)
    return _pct_ci([a[rng.integers(0, N, N)].mean() for _ in range(n_boot)])


def evaluate_classification(args, vocab):
    from sklearn.metrics import balanced_accuracy_score
    g2i = {g: i for i, g in enumerate(vocab)}
    df = pd.read_csv(Path(args.splits_dir) / "test_live.csv")
    df["path"] = df["path"].apply(lambda p: remap(p, args.photos_root))

    in_vocab = df["genus"].isin(g2i)
    skipped = int((~in_vocab).sum())
    df = df[in_vocab].reset_index(drop=True)
    if args.limit:
        df = df.sample(min(args.limit, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f"\n[CLASSIFICATION] test images: {len(df):,} "
          f"(skipped {skipped} in genera absent from model vocab)")

    loader = DataLoader(ImgDS(df["path"].tolist(), build_tfm(args.img_size, args._mean, args._std)),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    logits_raw = collect_logits(model, loader)

    # Pool synonymous output classes (confirmed taxonomy synonyms): sum the probability
    # of e.g. Trichotosia into Eria, so a model whose vocab still lists them separately is
    # comparable to one trained on the merged labels. Identity if vocab has no synonyms.
    syn = getattr(args, "_syn", {}) or {}
    mapped = [syn.get(g, g) for g in vocab]
    Mmat = None
    if set(mapped) != set(vocab):
        merged = sorted(set(mapped))
        mi = {g: i for i, g in enumerate(merged)}
        Mmat = np.zeros((len(vocab), len(merged)), dtype=np.float32)
        for j, g in enumerate(mapped):
            Mmat[j, mi[g]] = 1.0
        vocab, g2i = merged, mi
        print(f"  [synonyms] pooled {len(mapped) - len(merged)} class(es) -> {len(merged)} effective")

    def to_probs(lg):
        """raw logits -> (synonym-pooled) probabilities; mirrors the deployed softmax."""
        p = np_softmax(lg)
        return p @ Mmat if Mmat is not None else p

    probs = to_probs(logits_raw)
    labels = df["genus"].map(lambda g: g2i[syn.get(g, g)]).to_numpy()

    pred1 = probs.argmax(1)
    top5  = np.argsort(probs, 1)[:, -5:]
    correct1 = (pred1 == labels)
    correct5 = np.array([labels[i] in top5[i] for i in range(len(labels))])

    # per-genus accuracy
    per_genus = {}
    for gi in np.unique(labels):
        m = labels == gi
        per_genus[vocab[gi]] = {
            "support":  int(m.sum()),
            "top1_acc": float(correct1[m].mean()),
            "top5_acc": float(correct5[m].mean()),
        }
    macro_top1 = float(np.mean([v["top1_acc"] for v in per_genus.values()]))
    macro_top5 = float(np.mean([v["top5_acc"] for v in per_genus.values()]))

    ci = bootstrap_classification(labels, correct1, correct5, args.bootstrap, args.seed) \
        if args.bootstrap else {}

    results = {
        "n_test": len(df), "n_genera_evaluated": len(per_genus),
        "global_top1": float(correct1.mean()),
        "global_top5": float(correct5.mean()),
        "macro_top1": macro_top1,
        "macro_top5": macro_top5,
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred1)),
        "ece": expected_calibration_error(probs, labels),
        "bootstrap_n": args.bootstrap,
        "global_top1_ci95": ci.get("global_top1"),
        "macro_top1_ci95":  ci.get("macro_top1"),
        "macro_top5_ci95":  ci.get("macro_top5"),
    }
    def _ci(k):
        c = results.get(k)
        return f" [95% CI {c[0]*100:.1f}–{c[1]*100:.1f}]" if c else ""
    print(f"  Global  top1={results['global_top1']*100:.2f}%{_ci('global_top1_ci95')}  "
          f"top5={results['global_top5']*100:.2f}%")
    print(f"  Macro   top1={macro_top1*100:.2f}%{_ci('macro_top1_ci95')}  "
          f"top5={macro_top5*100:.2f}%{_ci('macro_top5_ci95')}  (global hides rare-genus failure)")
    print(f"  Balanced acc={results['balanced_accuracy']*100:.2f}%  ECE={results['ece']:.4f}")
    return results, per_genus, labels, pred1, vocab, logits_raw, to_probs


def fit_and_apply_temperature(args, model, vocab0, test_logits, test_labels, to_probs):
    """Temperature scaling (Guo et al., 2017): fit T on the val split, then report test ECE
    before/after. Top-1 / accuracy are invariant to T (monotonic), so only calibration moves."""
    g2i = {g: i for i, g in enumerate(vocab0)}
    vdf = pd.read_csv(Path(args.splits_dir) / args.temp_split)
    vdf["path"] = vdf["path"].apply(lambda p: remap(p, args.photos_root))
    vdf = vdf[vdf["genus"].isin(g2i)].reset_index(drop=True)
    if args.limit:
        vdf = vdf.sample(min(args.limit, len(vdf)), random_state=args.seed).reset_index(drop=True)
    print(f"\n[TEMPERATURE] fitting T on {args.temp_split}: {len(vdf):,} images")
    vloader = DataLoader(ImgDS(vdf["path"].tolist(), build_tfm(args.img_size, args._mean, args._std)),
                         batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, pin_memory=True)
    vlogits = collect_logits(model, vloader)
    vlabels = vdf["genus"].map(lambda g: g2i[g]).to_numpy()
    T = fit_temperature(vlogits, vlabels)

    ece_pre  = expected_calibration_error(to_probs(test_logits), test_labels)
    ece_post = expected_calibration_error(to_probs(test_logits / T), test_labels)
    acc_pre  = float((to_probs(test_logits).argmax(1) == test_labels).mean())
    acc_post = float((to_probs(test_logits / T).argmax(1) == test_labels).mean())
    print(f"  T*={T:.4f}   ECE {ece_pre:.4f} -> {ece_post:.4f}   "
          f"(top-1 {acc_pre*100:.2f}% -> {acc_post*100:.2f}%, unchanged)")
    return {"temperature": T, "ece_post_ts": ece_post, "ece_pre_ts": ece_pre,
            "temp_split": args.temp_split, "top1_acc_check": [acc_pre, acc_post]}


def save_classification_artifacts(out_dir, per_genus, labels, pred1, vocab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # per-genus CSV
    pg = pd.DataFrame([{"genus": g, **v} for g, v in per_genus.items()]) \
           .sort_values("top1_acc")
    pg.to_csv(out_dir / "per_genus_accuracy.csv", index=False)

    # per-genus bar
    fig, ax = plt.subplots(figsize=(10, max(6, len(pg) * 0.13)))
    ax.barh(pg["genus"], pg["top1_acc"], color=plt.cm.RdYlGn(pg["top1_acc"]))
    ax.set_xlabel("Top-1 accuracy"); ax.set_xlim(0, 1)
    ax.set_title("Per-genus top-1 accuracy (sorted)")
    ax.tick_params(axis="y", labelsize=5)
    fig.tight_layout(); fig.savefig(out_dir / "per_genus_accuracy.png", dpi=150); plt.close(fig)

    # confusion matrix (normalised over present labels)
    present = sorted(set(labels.tolist()) | set(pred1.tolist()))
    idx_map = {c: i for i, c in enumerate(present)}
    cm = np.zeros((len(present), len(present)), dtype=np.int64)
    for t, p in zip(labels, pred1):
        cm[idx_map[t], idx_map[p]] += 1
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    names = [vocab[c] for c in present]
    pd.DataFrame(cm, index=names, columns=names).to_csv(out_dir / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(max(8, len(present) * 0.12),) * 2)
    im = ax.imshow(cm_norm, cmap="magma", vmin=0, vmax=1)
    ax.set_title("Normalised confusion matrix"); fig.colorbar(im, fraction=0.046)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=150); plt.close(fig)

    # top-20 confused pairs (off-diagonal)
    pairs = []
    for i in range(len(present)):
        for j in range(len(present)):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": names[i], "pred": names[j],
                              "count": int(cm[i, j]),
                              "frac_of_true": float(cm_norm[i, j])})
    pd.DataFrame(sorted(pairs, key=lambda x: -x["count"])[:20]) \
        .to_csv(out_dir / "confused_pairs.csv", index=False)
    print(f"  Saved per-genus + confusion artifacts → {out_dir}")


# ── Retrieval (photo-level holdout) ───────────────────────────────────────────
def topk_cosine(ref, qry, k):
    """L2-normalised inner-product top-k. Uses FAISS if available, else numpy."""
    ref = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8)
    qry = qry / (np.linalg.norm(qry, axis=1, keepdims=True) + 1e-8)
    k = min(k, ref.shape[0])
    try:
        import faiss
        idx = faiss.IndexFlatIP(ref.shape[1]); idx.add(ref.astype("float32"))
        _, I = idx.search(qry.astype("float32"), k)
        return I
    except ImportError:
        sims = qry @ ref.T                       # (Q, R)
        return np.argsort(-sims, axis=1)[:, :k]


def evaluate_retrieval(args, model):
    df = pd.read_csv(Path(args.splits_dir) / "all_images.csv")
    df["path"] = df["path"].apply(lambda p: remap(p, args.photos_root))

    rng = random.Random(args.seed)
    by_species = defaultdict(list)
    for r in df.itertuples(index=False):
        by_species[r.species].append((r.path, r.genus))

    ref_paths, ref_sp, ref_gn = [], [], []
    qry_paths, qry_sp, qry_gn = [], [], []
    for sp, items in by_species.items():
        if len(items) >= args.min_photos:
            rng.shuffle(items)
            q = items[0]
            qry_paths.append(q[0]); qry_sp.append(sp); qry_gn.append(q[1])
            for p, g in items[1:]:
                ref_paths.append(p); ref_sp.append(sp); ref_gn.append(g)
        else:
            for p, g in items:
                ref_paths.append(p); ref_sp.append(sp); ref_gn.append(g)

    if args.limit:
        qn = max(1, args.limit // 5)
        qry_paths, qry_sp, qry_gn = qry_paths[:qn], qry_sp[:qn], qry_gn[:qn]
        ref_paths, ref_sp, ref_gn = ref_paths[:args.limit], ref_sp[:args.limit], ref_gn[:args.limit]

    print(f"\n[RETRIEVAL] reference={len(ref_paths):,}  queries={len(qry_paths):,}  "
          f"(species with >= {args.min_photos} photos)")

    tfm = build_tfm(args.img_size, args._mean, args._std)
    def embed(paths):
        loader = DataLoader(ImgDS(paths, tfm), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)
        _, emb = forward_all(model, loader, want_logits=False, want_emb=True)
        return emb.astype("float32")

    ref_emb = embed(ref_paths); qry_emb = embed(qry_paths)
    K = 10
    I = topk_cosine(ref_emb, qry_emb, K)
    ref_sp_arr = np.array(ref_sp); ref_gn_arr = np.array(ref_gn)

    hits_sp = {1: [], 5: [], 10: []}
    hits_gn = {1: [], 5: []}
    per_species_hit1 = defaultdict(list)
    for qi in range(len(qry_paths)):
        ret_sp = ref_sp_arr[I[qi]]
        ret_gn = ref_gn_arr[I[qi]]
        for k in (1, 5, 10):
            hits_sp[k].append(qry_sp[qi] in ret_sp[:k])
        for k in (1, 5):
            hits_gn[k].append(qry_gn[qi] in ret_gn[:k])
        per_species_hit1[qry_sp[qi]].append(qry_sp[qi] in ret_sp[:1])

    macro_r1 = float(np.mean([np.mean(v) for v in per_species_hit1.values()]))
    res = {
        "n_reference": len(ref_paths), "n_queries": len(qry_paths),
        "embed_dim": int(ref_emb.shape[1]),
        "species_recall@1":  float(np.mean(hits_sp[1])),
        "species_recall@5":  float(np.mean(hits_sp[5])),
        "species_recall@10": float(np.mean(hits_sp[10])),
        "species_recall@1_macro": macro_r1,
        "genus_recall@1": float(np.mean(hits_gn[1])),
        "genus_recall@5": float(np.mean(hits_gn[5])),
        "species_recall@5_ci95": bootstrap_mean(hits_sp[5], args.bootstrap, args.seed),
        "genus_recall@5_ci95":   bootstrap_mean(hits_gn[5], args.bootstrap, args.seed),
    }
    sp_ci = res["species_recall@5_ci95"]; gn_ci = res["genus_recall@5_ci95"]
    sp_ci_s = f" [95% CI {sp_ci[0]*100:.1f}–{sp_ci[1]*100:.1f}]" if sp_ci else ""
    gn_ci_s = f" [95% CI {gn_ci[0]*100:.1f}–{gn_ci[1]*100:.1f}]" if gn_ci else ""
    print(f"  Species  R@1={res['species_recall@1']*100:.2f}%  "
          f"R@5={res['species_recall@5']*100:.2f}%{sp_ci_s}  R@10={res['species_recall@10']*100:.2f}%  "
          f"(macro R@1={macro_r1*100:.2f}%)")
    print(f"  Genus    R@1={res['genus_recall@1']*100:.2f}%  R@5={res['genus_recall@5']*100:.2f}%{gn_ci_s}")
    return res


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    print(f"Device: {DEVICE} | Arch: {args.arch} | Model: {args.model} | img_size: {args.img_size}")

    vocab = json.loads(Path(args.vocab).read_text())
    safe_name = args.model.replace("/", "_").replace(":", "_")
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "eval" / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model, args._mean, args._std = load_model(args.arch, args.model, args.checkpoint,
                                               len(vocab), args.img_size)

    args._syn = load_synonyms(args.synonyms)
    if args._syn:
        print(f"Synonyms (confirmed): {args._syn}")

    vocab0 = vocab   # original (pre-synonym-pool) vocab order, for fitting T on raw logits
    cls_res, per_genus, labels, pred1, vocab, test_logits, to_probs = \
        evaluate_classification(args, vocab0)
    save_classification_artifacts(out_dir, per_genus, labels, pred1, vocab)

    if args.fit_temperature:
        cls_res.update(fit_and_apply_temperature(args, model, vocab0, test_logits, labels, to_probs))

    retr_res = None if args.no_retrieval else evaluate_retrieval(args, model)

    summary = {"model": args.model, "arch": args.arch, "img_size": args.img_size,
               "checkpoint": args.checkpoint,
               "classification": cls_res, "retrieval": retr_res}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary → {out_dir / 'results.json'}")
    print("Done.")
