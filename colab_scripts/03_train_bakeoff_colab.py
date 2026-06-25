"""
Backbone Bake-off Trainer — ONG Orchid Identifier v3  (Colab)
=============================================================
One parametric script to fine-tune & compare candidate backbones on an IDENTICAL
protocol, with all Phase-1 (imbalance) + Phase-2 (backbone/aug) improvements:

  * Backbone registry (pick with --model):
        effnetv2l   tf_efficientnetv2_l.in21k_ft_in1k        (timm, efficient baseline)
        convnextv2l convnextv2_large.fcmae_ft_in22k_in1k     (timm, best pure-ImageNet)
        dinov2l     vit_large_patch14_reg4_dinov2.lvd142m     (timm, best for retrieval)
        bioclip2    hf-hub:imageomics/bioclip-2               (open_clip, biology-pretrained — top pick)
  * Class imbalance:  WeightedRandomSampler (power-weighted) + Class-Balanced Focal loss
                      (Cui et al. 2019 effective number). --loss {cb_focal,ce}.
  * Augmentation:     RandomResizedCrop + flips + RandAugment (+ optional MixUp/CutMix).
  * Fine-tuning:      2-phase (head warm-up → full), discriminative LR (backbone < head),
                      warmup + cosine schedule, AMP, EMA weights.
  * Model selection:  best checkpoint by VAL **macro** top-1 (not global — global is
                      dominated by Bulbophyllum/Dendrobium).

Outputs to  <MODELS_DIR>/<model_key>/ :
    best_model.pth, vocab.json, training_history.csv, results.json

timm models are saved as NATIVE timm state_dicts, so they evaluate directly with
  notebooks/13_evaluate.py  --model <timm_name> --img-size <N> --vocab <...>
bioclip2 is saved as {backbone, head} (see BioClipClassifier) — eval needs the
  matching --arch openclip branch in 13_evaluate.py.

Colab setup (typical):
    from google.colab import drive; drive.mount('/content/drive')
    !pip -q install timm open_clip_torch
    # photos unzipped to /content/photos ; splits CSVs on Drive
    %run /content/drive/MyDrive/orchid_project/scripts/03_train_bakeoff_colab.py --model bioclip2

Run each candidate, then compare their eval/<model>/results.json.
"""

import argparse, json, time, csv, math, os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True   # dataset has a few slightly truncated JPEGs

# ── Backbone registry ─────────────────────────────────────────────────────────
REGISTRY = {
    "effnetv2l":   dict(arch="timm",     name="tf_efficientnetv2_l.in21k_ft_in1k",    img=448, bs=16),
    "convnextv2l": dict(arch="timm",     name="convnextv2_large.fcmae_ft_in22k_in1k", img=384, bs=16),
    "dinov2l":     dict(arch="timm",     name="vit_large_patch14_reg4_dinov2.lvd142m", img=448, bs=16),
    "bioclip2":    dict(arch="openclip", name="hf-hub:imageomics/bioclip-2",          img=224, bs=32),
}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def get_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--model", required=True, choices=list(REGISTRY),
                    help="which backbone to fine-tune")
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/orchid_project")
    ap.add_argument("--photos-root", default="/content/photos")
    ap.add_argument("--img-size",  type=int, default=0, help="override registry img size")
    ap.add_argument("--batch-size", type=int, default=0, help="override registry batch size")
    ap.add_argument("--workers",   type=int, default=2)
    ap.add_argument("--warmup-epochs", type=int, default=3, help="head-only warm-up epochs")
    ap.add_argument("--finetune-epochs", type=int, default=22, help="full fine-tune epochs")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--loss", choices=["cb_focal", "ce"], default="cb_focal")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--cb-beta", type=float, default=0.9999)
    ap.add_argument("--sampler-power", type=float, default=0.5,
                    help="class weight = 1/count**power (0=off, 0.5=sqrt, 1=inverse)")
    ap.add_argument("--mixup", action="store_true",
                    help="enable MixUp/CutMix (switches loss to soft-target CE)")
    ap.add_argument("--ema-decay", type=float, default=0.9998)
    ap.add_argument("--early-stop", type=int, default=8)
    ap.add_argument("--select", choices=["macro", "global"], default="macro",
                    help="metric driving best_model.pth + early-stop. BOTH "
                         "best_model_macro.pth and best_model_global.pth are ALWAYS saved "
                         "regardless, so one run yields both selection points.")
    ap.add_argument("--resume", action="store_true",
                    help="checkpoint+resume: save a full resume state (ckpt_resume.pth) to Drive "
                         "each epoch and, if one exists, continue from it instead of restarting. "
                         "Safe to always pass — first run starts fresh, a re-run after a crash "
                         "auto-continues from the last completed epoch.")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# ── Model wrappers ────────────────────────────────────────────────────────────
class BioClipClassifier(nn.Module):
    """open_clip visual encoder + linear head. .embed() returns image features."""
    def __init__(self, open_clip_model, embed_dim, n_cls):
        super().__init__()
        self.visual = open_clip_model.visual
        self.head = nn.Linear(embed_dim, n_cls)
    def embed(self, x):
        return self.visual(x)
    def forward(self, x):
        return self.head(self.embed(x))


def build_model(spec, n_cls, img_size):
    """Returns (model, mean, std). timm → native model; openclip → BioClipClassifier."""
    if spec["arch"] == "timm":
        import timm
        kwargs = dict(pretrained=True, num_classes=n_cls)
        if "vit" in spec["name"]:
            kwargs["img_size"] = img_size          # interpolate ViT pos-embed
        model = timm.create_model(spec["name"], **kwargs)
        cfg = timm.data.resolve_model_data_config(model)
        return model, cfg["mean"], cfg["std"]
    else:  # open_clip / BioCLIP-2
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms(spec["name"])
        embed_dim = clip_model.visual.output_dim
        model = BioClipClassifier(clip_model, embed_dim, n_cls)
        return model, CLIP_MEAN, CLIP_STD


def model_embed(model, x):
    """Pre-logits embedding for either wrapper type."""
    if isinstance(model, BioClipClassifier):
        return model.embed(x)
    feat = model.forward_features(x)
    return model.forward_head(feat, pre_logits=True)


def set_backbone_frozen(model, frozen: bool):
    if isinstance(model, BioClipClassifier):
        for p in model.visual.parameters():
            p.requires_grad = not frozen
        for p in model.head.parameters():
            p.requires_grad = True
    else:
        clf = model.get_classifier()
        for p in model.parameters():
            p.requires_grad = not frozen
        for p in clf.parameters():
            p.requires_grad = True


def param_groups(model, lr_head, lr_backbone, wd):
    if isinstance(model, BioClipClassifier):
        head_params = list(model.head.parameters())
        back_params = list(model.visual.parameters())
    else:
        clf = model.get_classifier()
        head_ids = {id(p) for p in clf.parameters()}
        head_params = [p for p in model.parameters() if id(p) in head_ids]
        back_params = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": head_params, "lr": lr_head, "weight_decay": wd},
        {"params": back_params, "lr": lr_backbone, "weight_decay": wd},
    ]


# ── Loss ──────────────────────────────────────────────────────────────────────
class CBFocalLoss(nn.Module):
    """Class-Balanced Focal Loss (Cui et al. 2019, effective number of samples)."""
    def __init__(self, samples_per_cls, beta=0.9999, gamma=2.0):
        super().__init__()
        eff = 1.0 - np.power(beta, np.asarray(samples_per_cls, dtype=np.float64))
        w = (1.0 - beta) / np.maximum(eff, 1e-8)
        w = w / w.sum() * len(samples_per_cls)
        self.register_buffer("weight", torch.tensor(w, dtype=torch.float32))
        self.gamma = gamma
    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, target, reduction="none")
        pt = logp.gather(1, target[:, None]).squeeze(1).exp()
        focal = (1.0 - pt) ** self.gamma * ce
        return (self.weight[target] * focal).mean()


# ── Data ──────────────────────────────────────────────────────────────────────
def remap(path, photos_root):
    parts = path.replace("\\", "/").split("/")
    low = [p.lower() for p in parts]
    if "photos" in low:
        i = low.index("photos")
        return photos_root.rstrip("/") + "/" + "/".join(parts[i + 1:])
    return path


class OrchidDS(Dataset):
    def __init__(self, df, g2i, tfm):
        self.df, self.g2i, self.tfm = df.reset_index(drop=True), g2i, tfm
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.tfm(img), self.g2i[r["genus"]]


def build_transforms(img_size, mean, std):
    train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25),
    ])
    val = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train, val


# ── Eval ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, n_cls):
    model.eval()
    all_p, all_l = [], []
    for imgs, labs in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model(imgs.to(device))
        all_p.append(out.float().cpu()); all_l.append(labs)
    p = torch.cat(all_p); l = torch.cat(all_l).numpy()
    pred1 = p.argmax(1).numpy()
    top5 = p.topk(5, dim=1).indices.numpy()
    g_top1 = float((pred1 == l).mean())
    g_top5 = float(np.mean([l[i] in top5[i] for i in range(len(l))]))
    # macro top-1 over genera present in this split
    macro = []
    for c in np.unique(l):
        m = l == c
        macro.append((pred1[m] == c).mean())
    return g_top1, g_top5, float(np.mean(macro))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = REGISTRY[args.model]
    img_size = args.img_size or spec["img"]
    batch_size = args.batch_size or spec["bs"]

    drive = Path(args.drive_root)
    splits_dir = drive / "data" / "splits"
    out_dir = drive / "models" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device} | Model: {args.model} ({spec['name']}) | "
          f"img={img_size} | bs={batch_size}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # data
    train_df = pd.read_csv(splits_dir / "train_live.csv")
    val_df   = pd.read_csv(splits_dir / "val_live.csv")
    for d in (train_df, val_df):
        d["path"] = d["path"].apply(lambda p: remap(p, args.photos_root))
    genera = sorted(train_df["genus"].unique())
    g2i = {g: i for i, g in enumerate(genera)}
    n_cls = len(genera)
    val_df = val_df[val_df["genus"].isin(g2i)].reset_index(drop=True)
    print(f"Genera: {n_cls} | Train: {len(train_df):,} | Val: {len(val_df):,}")
    json.dump(genera, open(out_dir / "vocab.json", "w"), indent=2)

    # model + transforms
    model, mean, std = build_model(spec, n_cls, img_size)
    model = model.to(device)
    train_tfm, val_tfm = build_transforms(img_size, mean, std)

    # weighted sampler (class-balanced sampling)
    counts = np.array([(train_df["genus"] == g).sum() for g in genera], dtype=np.float64)
    if args.sampler_power > 0:
        cls_w = 1.0 / np.power(counts, args.sampler_power)
        sample_w = train_df["genus"].map(lambda g: cls_w[g2i[g]]).to_numpy()
        sampler = WeightedRandomSampler(torch.tensor(sample_w, dtype=torch.double),
                                        num_samples=len(train_df), replacement=True)
        shuffle = False
    else:
        sampler, shuffle = None, True

    train_dl = DataLoader(OrchidDS(train_df, g2i, train_tfm), batch_size=batch_size,
                          sampler=sampler, shuffle=shuffle, num_workers=args.workers,
                          pin_memory=True, drop_last=True)
    val_dl   = DataLoader(OrchidDS(val_df, g2i, val_tfm), batch_size=batch_size,
                          shuffle=False, num_workers=args.workers, pin_memory=True)

    # mixup (optional)
    mixup_fn = None
    if args.mixup:
        from timm.data import Mixup
        mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5,
                         switch_prob=0.5, label_smoothing=0.1, num_classes=n_cls)
        from timm.loss import SoftTargetCrossEntropy
        criterion = SoftTargetCrossEntropy()
        print("MixUp/CutMix ON → soft-target CE (class-balanced loss disabled)")
    elif args.loss == "cb_focal":
        criterion = CBFocalLoss(counts, beta=args.cb_beta, gamma=args.focal_gamma).to(device)
        print(f"Loss: Class-Balanced Focal (beta={args.cb_beta}, gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        print("Loss: CrossEntropy (label_smoothing=0.1)")

    # EMA
    from timm.utils import ModelEmaV3
    ema = ModelEmaV3(model, decay=args.ema_decay)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    total_epochs = args.warmup_epochs + args.finetune_epochs

    def make_opt_sched(epochs, lr_head, lr_backbone):
        opt = torch.optim.AdamW(param_groups(model, lr_head, lr_backbone, args.weight_decay))
        warm = max(1, int(0.05 * epochs) + 1)
        def lr_lambda(ep):
            if ep < warm:
                return (ep + 1) / warm
            prog = (ep - warm) / max(1, epochs - warm)
            return 0.5 * (1 + math.cos(math.pi * prog))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return opt, sched

    history, best_macro, best_global, patience = [], 0.0, 0.0, 0

    # ── Resume: load a previous per-epoch checkpoint from Drive (if present) ─────
    resume_info = None
    if args.resume:
        ckpt_path = out_dir / "ckpt_resume.pth"
        if ckpt_path.exists():
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"]); ema.module.load_state_dict(ck["ema"])
            best_macro, best_global = ck["best_macro"], ck["best_global"]
            patience, history = ck["patience"], ck["history"]
            resume_info = (ck["phase"], ck["epoch"])
            print(f"RESUME: loaded {ckpt_path.name} — phase={ck['phase']} epoch={ck['epoch']} | "
                  f"best_macro={best_macro*100:.2f}% best_global={best_global*100:.2f}% | "
                  f"{len(history)} epochs of history restored. Continuing from here.")
        else:
            print("RESUME on, but no ckpt_resume.pth yet — starting fresh; will checkpoint each epoch.")

    def save_resume(phase, ep):
        """Atomically persist a full resume state to Drive after each epoch.
        Stores model+EMA weights, schedule position (phase/epoch), best metrics, patience, and
        history. On resume the optimizer momentum & AMP scaler restart fresh (a negligible
        perturbation for fine-tuning) while weights/EMA/LR-schedule continue exactly."""
        if not args.resume:
            return
        tmp, final = out_dir / "ckpt_resume.tmp.pth", out_dir / "ckpt_resume.pth"
        torch.save(dict(phase=phase, epoch=ep,
                        model=model.state_dict(), ema=ema.module.state_dict(),
                        best_macro=best_macro, best_global=best_global,
                        patience=patience, history=history, args=vars(args)), tmp)
        os.replace(tmp, final)   # never leave a half-written checkpoint if it crashes mid-save

    def run_phase(name, epochs, freeze_backbone, lr_head, lr_backbone, start_ep=1):
        nonlocal best_macro, best_global, patience
        set_backbone_frozen(model, freeze_backbone)
        trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'='*60}\nPhase {name} | epochs={epochs} | "
              f"lr_head={lr_head} lr_bb={lr_backbone} | trainable={trn:,}")
        opt, sched = make_opt_sched(epochs, lr_head, lr_backbone)
        for _ in range(start_ep - 1):     # fast-forward the LR schedule when resuming
            sched.step()
        if start_ep > 1:
            print(f"  >> resuming '{name}' at epoch {start_ep}/{epochs} "
                  f"(weights/EMA/LR-schedule restored; optimizer & AMP scaler start fresh)")
        for ep in range(start_ep, epochs + 1):
            model.train(); t0 = time.time(); tot = 0.0
            for imgs, labs in tqdm(train_dl, leave=False):
                imgs, labs = imgs.to(device), labs.to(device)
                if mixup_fn is not None:
                    imgs, labs = mixup_fn(imgs, labs)
                opt.zero_grad()
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    loss = criterion(model(imgs), labs)
                if scaler:
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
                ema.update(model)
                tot += loss.item() * imgs.size(0)
            sched.step()
            g1, g5, mac = evaluate(model, val_dl, device, n_cls)
            e1, e5, emac = evaluate(ema.module, val_dl, device, n_cls)
            sel_mac  = max(mac, emac); use_ema_mac  = emac > mac   # best macro (raw vs ema)
            sel_glob = max(g1, e1);    use_ema_glob = e1 > g1       # best global (raw vs ema)
            print(f"  [{name} {ep}/{epochs}] loss={tot/len(train_df):.4f} "
                  f"val_macro={mac*100:.2f}% (ema {emac*100:.2f}%) "
                  f"val_top1={g1*100:.2f}% (ema {e1*100:.2f}%) top5={g5*100:.2f}% {time.time()-t0:.0f}s")
            history.append(dict(phase=name, epoch=ep, loss=round(tot/len(train_df), 4),
                                val_macro=round(mac, 4), val_ema_macro=round(emac, 4),
                                val_top1=round(g1, 4), val_ema_top1=round(e1, 4), val_top5=round(g5, 4)))
            # Always persist BOTH selection points; best_model.pth mirrors --select.
            saved_primary = False
            if sel_mac > best_macro:
                best_macro = sel_mac
                sd = (ema.module if use_ema_mac else model).state_dict()
                torch.save(sd, out_dir / "best_model_macro.pth")
                print(f"    >> best val_macro={best_macro*100:.2f}% "
                      f"({'EMA' if use_ema_mac else 'raw'}) — saved")
                if args.select == "macro":
                    torch.save(sd, out_dir / "best_model.pth"); patience = 0; saved_primary = True
            if sel_glob > best_global:
                best_global = sel_glob
                sd = (ema.module if use_ema_glob else model).state_dict()
                torch.save(sd, out_dir / "best_model_global.pth")
                print(f"    >> best val_top1={best_global*100:.2f}% "
                      f"({'EMA' if use_ema_glob else 'raw'}) — saved")
                if args.select == "global":
                    torch.save(sd, out_dir / "best_model.pth"); patience = 0; saved_primary = True
            if not saved_primary:
                patience += 1
            save_resume(name, ep)            # checkpoint after every epoch (no-op unless --resume)
            if not saved_primary and patience >= args.early_stop:
                print("    >> early stop"); return True
        return False

    def finetune(start):
        return run_phase("finetune", args.finetune_epochs, False,
                         args.lr_head * 0.3, args.lr_backbone, start_ep=start)
    if resume_info and resume_info[0] == "finetune":
        finetune(resume_info[1] + 1)                       # crashed during fine-tune → resume there
    elif resume_info and resume_info[1] >= args.warmup_epochs:
        finetune(1)                                        # warm-up had finished → start fine-tune
    else:                                                  # fresh run, or crashed during warm-up
        wstart = (resume_info[1] + 1) if resume_info else 1
        stop = run_phase("warmup", args.warmup_epochs, True, args.lr_head, 0.0, start_ep=wstart)
        if not stop:
            finetune(1)

    # save history + results
    with open(out_dir / "training_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)
    results = dict(model=args.model, timm_name=spec["name"], arch=spec["arch"],
                   img_size=img_size, genera=n_cls, train=len(train_df), val=len(val_df),
                   select=args.select, loss=("mixup" if args.mixup else args.loss),
                   sampler_power=args.sampler_power,
                   best_val_macro=round(best_macro * 100, 2),
                   best_val_top1=round(best_global * 100, 2))
    json.dump(results, open(out_dir / "results.json", "w"), indent=2)
    print(f"\nDONE. best_val_macro={best_macro*100:.2f}% | best_val_top1={best_global*100:.2f}% "
          f"| --select={args.select} → best_model.pth | saved → {out_dir}")
    print("Checkpoints: best_model_macro.pth, best_model_global.pth (eval BOTH on the test set).")
    if args.resume:
        (out_dir / "ckpt_resume.pth").unlink(missing_ok=True)
        print("Resume checkpoint cleared — training completed successfully.")
    print("Next: evaluate on test with notebooks/13_evaluate.py, then build FAISS (04).")


if __name__ == "__main__":
    main()
