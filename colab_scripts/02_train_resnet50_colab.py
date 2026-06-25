"""
ResNet50 Baseline Genus Classifier — Colab Version
Run via: %run /content/drive/MyDrive/orchid_project/scripts/02_train_resnet50_colab.py

Uses FastAI (same approach as v1 Bird's Head Classifier).
Model saved to Google Drive for persistence.
"""

from fastai.vision.all import *
import json
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SPLITS_DIR  = Path("/content/drive/MyDrive/orchid_project/data/splits")
MODELS_DIR  = Path("/content/drive/MyDrive/orchid_project/models/resnet50")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE        = 224
BATCH_SIZE      = 64     # T4 16 GB — can go larger than laptop
FREEZE_EPOCHS   = 3
FINETUNE_EPOCHS = 15
SEED            = 42

set_seed(SEED)

# ── Load data ─────────────────────────────────────────────────────────────────
train_df = pd.read_csv(SPLITS_DIR / "train_live.csv")
val_df   = pd.read_csv(SPLITS_DIR / "val_live.csv")
test_df  = pd.read_csv(SPLITS_DIR / "test_live.csv")

# Remap paths from local Windows paths to Colab /content/photos/
def remap_path(p: str) -> str:
    # Normalise separators: Windows backslash → forward slash
    parts = p.replace("\\", "/").split("/")
    try:
        idx = [x.lower() for x in parts].index("photos")
        return "/content/" + "/".join(parts[idx:])
    except ValueError:
        return p

train_df["path"] = train_df["path"].apply(remap_path)
val_df["path"]   = val_df["path"].apply(remap_path)
test_df["path"]  = test_df["path"].apply(remap_path)

train_df["is_valid"] = False
val_df["is_valid"]   = True
df = pd.concat([train_df, val_df], ignore_index=True)

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Genera: {df['genus'].nunique()}")

dls = ImageDataLoaders.from_df(
    df, path="/", fn_col="path", label_col="genus", valid_col="is_valid",
    item_tfms=Resize(IMG_SIZE),
    batch_tfms=aug_transforms(mult=2.0, do_flip=True),
    bs=BATCH_SIZE,
)

# ── Train ─────────────────────────────────────────────────────────────────────
learn = vision_learner(dls, resnet50, metrics=[accuracy, RocAuc(average="macro")],
                       model_dir=str(MODELS_DIR))
print("\nFinding optimal LR...")
suggested = learn.lr_find(suggest_funcs=(valley, slide))
print(f"Suggested LR: {suggested}")

print(f"\nTraining: {FREEZE_EPOCHS} frozen + {FINETUNE_EPOCHS} fine-tune epochs...")
learn.fine_tune(FINETUNE_EPOCHS, freeze_epochs=FREEZE_EPOCHS)

# ── Save ──────────────────────────────────────────────────────────────────────
learn.export(MODELS_DIR / "resnet50_export.pkl")
vocab = list(dls.vocab)
with open(MODELS_DIR / "vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)
print(f"\nSaved to Drive: {MODELS_DIR}")

# ── Test evaluation ───────────────────────────────────────────────────────────
test_dl  = dls.test_dl(test_df["path"].tolist(), num_workers=2)
preds, _ = learn.get_preds(dl=test_dl)
labels   = tensor([dls.vocab.o2i.get(g, 0) for g in test_df["genus"]])

top1 = (preds.argmax(1) == labels).float().mean().item()
top5 = sum(1 for p, t in zip(preds, labels) if t in p.topk(5).indices) / len(labels)
print(f"\nTest Top-1: {top1*100:.2f}%  |  Top-5: {top5*100:.2f}%")

results = {"model":"ResNet50","test_top1":round(top1*100,2),"test_top5":round(top5*100,2),
           "genera":dls.c,"train":len(train_df),"val":len(val_df),"test":len(test_df)}
with open(MODELS_DIR / "results.json","w") as f:
    json.dump(results, f, indent=2)
print("Done. Run next: 03_train_efficientnet_colab.py")
