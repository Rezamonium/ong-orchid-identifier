"""
EfficientNet-B4 Genus Classifier — Colab Version
Run via: %run /content/drive/MyDrive/orchid_project/scripts/03_train_efficientnet_colab.py

3-phase progressive unfreezing. T4 GPU (16 GB) allows batch_size=32 at 380px.
Best model checkpoint saved to Google Drive automatically.
"""

import json, time, csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# ── Paths ─────────────────────────────────────────────────────────────────────
SPLITS_DIR  = Path("/content/drive/MyDrive/orchid_project/data/splits")
MODELS_DIR  = Path("/content/drive/MyDrive/orchid_project/models/efficientnet_b4")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = 380
BATCH_SIZE  = 32    # T4 16 GB — safe for B4 at 380px
SEED        = 42
NUM_WORKERS = 2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PHASES = [
    {"name": "head_only",        "epochs": 5,  "lr": 1e-3, "unfreeze": 0},
    {"name": "partial_unfreeze", "epochs": 10, "lr": 1e-4, "unfreeze": 2},
    {"name": "full_finetune",    "epochs": 15, "lr": 5e-5, "unfreeze": -1},
]
EARLY_STOP  = 7
LABEL_SMOOTH = 0.1

torch.manual_seed(SEED); np.random.seed(SEED)
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Dataset ───────────────────────────────────────────────────────────────────
class OrchidDS(Dataset):
    def __init__(self, df, g2i, tfm):
        self.df, self.g2i, self.tfm = df.reset_index(drop=True), g2i, tfm
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.tfm(img), self.g2i[r["genus"]]

train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tfm = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.1)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ── Remap Windows paths → Colab /content/photos/ ──────────────────────────────
def remap_path(p: str) -> str:
    # Normalise separators: Windows backslash → forward slash
    parts = p.replace("\\", "/").split("/")
    try:
        idx = [x.lower() for x in parts].index("photos")
        return "/content/" + "/".join(parts[idx:])
    except ValueError:
        return p

train_df = pd.read_csv(SPLITS_DIR/"train_live.csv")
val_df   = pd.read_csv(SPLITS_DIR/"val_live.csv")
test_df  = pd.read_csv(SPLITS_DIR/"test_live.csv")

train_df["path"] = train_df["path"].apply(remap_path)
val_df["path"]   = val_df["path"].apply(remap_path)
test_df["path"]  = test_df["path"].apply(remap_path)

# Filter val/test to only genera present in train (avoids KeyError for rare genera)
train_genera = set(train_df["genus"].unique())
val_df   = val_df[val_df["genus"].isin(train_genera)]
test_df  = test_df[test_df["genus"].isin(train_genera)]

genera   = sorted(train_genera)
g2i      = {g:i for i,g in enumerate(genera)}
i2g      = {i:g for g,i in g2i.items()}
n_cls    = len(genera)
print(f"Genera: {n_cls} | Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

with open(MODELS_DIR/"vocab.json","w") as f: json.dump(genera,f,indent=2)

def make_loader(df, tfm, shuffle):
    return DataLoader(OrchidDS(df,g2i,tfm), batch_size=BATCH_SIZE,
                      shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=True)

train_dl = make_loader(train_df, train_tfm, True)
val_dl   = make_loader(val_df,   val_tfm,   False)
test_dl  = make_loader(test_df,  val_tfm,   False)

# ── Model ─────────────────────────────────────────────────────────────────────
model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=n_cls).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
scaler    = torch.amp.GradScaler("cuda") if DEVICE.type=="cuda" else None

def set_freeze(model, unfreeze):
    for p in model.parameters(): p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = True
    if unfreeze == -1:
        for p in model.parameters(): p.requires_grad = True
    elif unfreeze > 0:
        for blk in list(model.blocks)[-unfreeze:]:
            for p in blk.parameters(): p.requires_grad = True
        for p in model.conv_head.parameters(): p.requires_grad = True

def run_epoch(loader, train=True, optimizer=None):
    model.train() if train else model.eval()
    tot_loss = correct = total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labs in tqdm(loader, leave=False):
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            if train: optimizer.zero_grad()
            if scaler and train:
                with torch.amp.autocast("cuda"):
                    out  = model(imgs); loss = criterion(out, labs)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                out  = model(imgs); loss = criterion(out, labs)
                if train: loss.backward(); optimizer.step()
            tot_loss += loss.item()*imgs.size(0)
            correct  += (out.argmax(1)==labs).sum().item()
            total    += imgs.size(0)
    return tot_loss/total, correct/total

def evaluate_topk(loader):
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            all_p.append(model(imgs.to(DEVICE)).cpu())
            all_l.append(labs)
    p = torch.cat(all_p); l = torch.cat(all_l).numpy()
    top1 = accuracy_score(l, p.argmax(1).numpy())
    top5 = top_k_accuracy_score(l, p.numpy(), k=5)
    loss = criterion(p, torch.tensor(l, dtype=torch.long)).item()
    return top1, top5, loss

# ── Training loop ─────────────────────────────────────────────────────────────
history = []; best_val = 0.0; patience = 0

for phase in PHASES:
    set_freeze(model, phase["unfreeze"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*55}\nPhase: {phase['name']} | Epochs: {phase['epochs']} | "
          f"LR: {phase['lr']} | Trainable params: {trainable:,}")

    opt  = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=phase["lr"], weight_decay=1e-4)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=phase["epochs"])

    for ep in range(1, phase["epochs"]+1):
        t0 = time.time()
        tr_loss, tr_acc    = run_epoch(train_dl, True,  opt)
        v1, v5, val_loss   = evaluate_topk(val_dl)
        sch.step()

        print(f"  [{phase['name']} {ep}/{phase['epochs']}]  "
              f"loss={tr_loss:.4f}  val_top1={v1*100:.2f}%  "
              f"val_top5={v5*100:.2f}%  {time.time()-t0:.0f}s")

        history.append({"phase":phase["name"],"epoch":ep,
                        "train_loss":round(tr_loss,4),"train_acc":round(tr_acc,4),
                        "val_loss":round(val_loss,4),"val_top1":round(v1,4),"val_top5":round(v5,4)})

        if v1 > best_val:
            best_val = v1; patience = 0
            torch.save(model.state_dict(), MODELS_DIR/"best_model.pth")
            print(f"    >> Best val: {best_val*100:.2f}% — saved to Drive")
        else:
            patience += 1
            if patience >= EARLY_STOP:
                print(f"    >> Early stopping"); break

# ── Save history ──────────────────────────────────────────────────────────────
with open(MODELS_DIR/"training_history.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)

# ── Test evaluation ───────────────────────────────────────────────────────────
model.load_state_dict(torch.load(MODELS_DIR/"best_model.pth", map_location=DEVICE))
t1, t5, tloss = evaluate_topk(test_dl)
print(f"\n{'='*55}")
print(f"TEST RESULTS — Best model")
print(f"  Top-1: {t1*100:.2f}%  |  Top-5: {t5*100:.2f}%  |  Loss: {tloss:.4f}")

results = {"model":"EfficientNet-B4","best_val_top1":round(best_val*100,2),
           "test_top1":round(t1*100,2),"test_top5":round(t5*100,2),
           "genera":n_cls,"train":len(train_df),"val":len(val_df),"test":len(test_df)}
with open(MODELS_DIR/"results.json","w") as f: json.dump(results,f,indent=2)
print(f"Saved: {MODELS_DIR}")
print("Done. Run next: 04_build_faiss_colab.py")
