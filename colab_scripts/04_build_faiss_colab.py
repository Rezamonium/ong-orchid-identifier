"""
FAISS Similarity Index Builder — Colab Version
Run via: %run /content/drive/MyDrive/orchid_project/scripts/04_build_faiss_colab.py

Loads trained EfficientNet-B4, strips the head, extracts 1792-dim embeddings.
Filters:
  - Only live/uncertain photos (from image_type_labels_2ndrev.csv)
  - Excludes _NID (iNaturalist Needs-ID) — unconfirmed species labels
Species metadata loaded from bh_species_map.json (BH, KM, iNat photos).
Builds a cosine-similarity FAISS index. Index + metadata saved to Google Drive.
"""

import json
import torch
import torch.nn as nn
import numpy as np
try:
    import faiss
except ImportError:
    import subprocess, sys
    print("faiss not found — installing faiss-cpu ...")
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "faiss-cpu"], check=True)
    import faiss
import timm
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
PHOTOS_DIR   = Path("/content/photos")
LABELS_CSV   = Path("/content/drive/MyDrive/orchid_project/data/image_type_labels_2ndrev.csv")
SP_MAP_JSON  = Path("/content/drive/MyDrive/orchid_project/data/bh_species_map.json")
DRIVE_MODELS = Path("/content/drive/MyDrive/orchid_project/models")
EFFNET_DIR   = DRIVE_MODELS / "efficientnet_b4"

IMG_SIZE     = 380
BATCH_SIZE   = 64
VALID_EXTS   = {".jpg",".jpeg",".png",".bmp",".tiff",".tif",".webp"}
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

# ── Load model ────────────────────────────────────────────────────────────────
with open(EFFNET_DIR/"vocab.json") as f: genera = json.load(f)
n_cls = len(genera)

model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=n_cls)
model.load_state_dict(torch.load(EFFNET_DIR/"best_model.pth", map_location=DEVICE))
model.classifier = nn.Identity()   # 1792-dim embedding output
model = model.to(DEVICE).eval()
print(f"Model loaded (embedding dim=1792, genera={n_cls})")

# ── Transform ─────────────────────────────────────────────────────────────────
tfm = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.1)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ── Load species map (BH, KM, iNat filenames → species name) ─────────────────
import csv as _csv
sp_lookup = {}
if SP_MAP_JSON.exists():
    with open(SP_MAP_JSON, encoding="utf-8") as f:
        sp_lookup = json.load(f)
    print(f"Species map loaded: {len(sp_lookup):,} entries")
else:
    print("WARNING: bh_species_map.json not found — species metadata will be parsed from filenames")

def parse_species(filename: str) -> str:
    """Extract species from filename; BH/KM/iNat use sp_lookup."""
    if filename in sp_lookup:
        return sp_lookup[filename]
    # ONG / JC: {Genus}_{Genus}_{epithet}_{seq}[_Attribution]
    stem  = Path(filename).stem
    parts = stem.split("_")
    while parts and not parts[-1].isdigit():
        parts = parts[:-1]
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts[1:])

# ── Load live-only labels (exclude illustrations, herbarium & NID) ────────────
def path_to_filename(p: str) -> str:
    """Extract filename from a Windows or Linux path string."""
    return p.replace("\\", "/").split("/")[-1]

live_paths = set()
if LABELS_CSV.exists():
    with open(LABELS_CSV, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            if row["label"] in ("live", "uncertain"):
                fn = path_to_filename(row["path"])
                if "_NID" not in fn:   # exclude unconfirmed iNat Needs-ID
                    live_paths.add(fn)
    print(f"Live filter loaded: {len(live_paths):,} photos (NID excluded)")
else:
    print("WARNING: image_type_labels_2ndrev.csv not found — indexing ALL non-NID photos")

# ── Collect images ────────────────────────────────────────────────────────────
records = []
for genus_dir in sorted(PHOTOS_DIR.iterdir()):
    if not genus_dir.is_dir(): continue
    genus = genus_dir.name
    for f in genus_dir.iterdir():
        if not (f.is_file() and f.suffix.lower() in VALID_EXTS):
            continue
        if "_NID" in f.name:
            continue   # always exclude NID from FAISS
        if live_paths and f.name not in live_paths:
            continue   # skip illustrations & herbarium
        records.append({
            "path":    str(f),
            "genus":   genus,
            "species": parse_species(f.name),
            "filename": f.name,
        })

print(f"Images to index: {len(records):,}")

# ── Extract embeddings ────────────────────────────────────────────────────────
embeddings, valid_recs = [], []
batch_imgs, batch_recs = [], []

def flush_batch():
    if not batch_imgs: return
    with torch.no_grad():
        t = torch.stack(batch_imgs).to(DEVICE)
        e = model(t).cpu().numpy()
    embeddings.extend(e)
    valid_recs.extend(batch_recs)
    batch_imgs.clear(); batch_recs.clear()

for rec in tqdm(records, desc="Extracting"):
    try:
        img = Image.open(rec["path"]).convert("RGB")
        batch_imgs.append(tfm(img))
        batch_recs.append(rec)
    except Exception as ex:
        print(f"  Skipping {rec['path']}: {ex}")
        continue
    if len(batch_imgs) == BATCH_SIZE:
        flush_batch()
flush_batch()

print(f"Embeddings: {len(embeddings):,}")

# ── Build FAISS index ─────────────────────────────────────────────────────────
arr = np.array(embeddings, dtype="float32")
faiss.normalize_L2(arr)   # L2-normalise → cosine sim = inner product

idx = faiss.IndexFlatIP(arr.shape[1])
idx.add(arr)
print(f"FAISS index: {idx.ntotal:,} vectors, dim={arr.shape[1]}")

# ── Save to Drive ─────────────────────────────────────────────────────────────
faiss.write_index(idx, str(DRIVE_MODELS/"ong_species_index.faiss"))
with open(DRIVE_MODELS/"ong_metadata.json","w",encoding="utf-8") as f:
    json.dump(valid_recs, f, ensure_ascii=False)

print(f"Saved: {DRIVE_MODELS/'ong_species_index.faiss'}")
print(f"Saved: {DRIVE_MODELS/'ong_metadata.json'}")

# ── Sanity check ──────────────────────────────────────────────────────────────
print("\nSanity check (first image):")
D, I = idx.search(arr[:1], k=6)
for rank, (score, i) in enumerate(zip(D[0], I[0]), 1):
    print(f"  Top-{rank}: {valid_recs[i]['species']}  (score={score:.4f})")

print("\nDone. Run next: 05_gradio_app_colab.py  (or test_retrieval)")
