"""
Dataset Reorganisation — ONG Orchid Identifier v2
Day 1-2: Reorganise species folders into genus-level folders for classifier training.

Actions:
  1. Skip/exclude the 100 empty species folders
  2. Fix the "AgrostophylIum" typo (capital I) -> "Agrostophyllum"
  3. Copy images into genus_classifier/<Genus>/<species>/<filename>
     (preserves species sub-folder for traceability)
  4. Generate stratified 70/15/15 train/val/test split CSVs
  5. Print final stats

Output:
  E:/Claude Code/ONG_v2/data/genus_classifier/<Genus>/<species>/<img>
  E:/Claude Code/ONG_v2/data/splits/train.csv
  E:/Claude Code/ONG_v2/data/splits/val.csv
  E:/Claude Code/ONG_v2/data/splits/test.csv
  E:/Claude Code/ONG_v2/data/splits/all_images.csv
"""

import os
import csv
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ── Config ───────────────────────────────────────────────────────────────────
PHOTOS_DIR      = Path("E:/Claude Code/scrapping ONG/photos")
GENUS_DIR       = Path("E:/Claude Code/ONG_v2/data/genus_classifier")
SPLITS_DIR      = Path("E:/Claude Code/ONG_v2/data/splits")
VALID_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
TRAIN_FRAC      = 0.70
VAL_FRAC        = 0.15
TEST_FRAC       = 0.15
RANDOM_SEED     = 42
MIN_IMAGES_FOR_GENUS_TRAIN = 20   # genera below this are noted but still included
# ─────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)
GENUS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Typo fix map  (source folder name prefix → corrected genus)
GENUS_FIXES = {
    "AgrostophylIum": "Agrostophyllum",  # capital I -> lowercase l
}

def get_genus(species_name: str) -> str:
    """Extract genus from species name, applying known fixes."""
    raw = species_name.split()[0]
    return GENUS_FIXES.get(raw, raw)


# ── Step 1: Collect all valid images ─────────────────────────────────────────
print("Step 1: Collecting images from source...")
all_records = []   # {img_path, species, genus, rel_path}
skipped_empty = 0

species_dirs = sorted([d for d in PHOTOS_DIR.iterdir() if d.is_dir()])
for sp_dir in species_dirs:
    species_name = sp_dir.name
    genus        = get_genus(species_name)
    img_files    = [f for f in sp_dir.iterdir()
                    if f.is_file()
                    and f.suffix.lower() in VALID_EXTS
                    and not f.name.startswith("Distribution of")]
    if not img_files:
        skipped_empty += 1
        continue
    for img_path in img_files:
        all_records.append({
            "src_path":    img_path,
            "species":     species_name,
            "genus":       genus,
        })

print(f"  Found {len(all_records):,} images across species ({skipped_empty} empty folders skipped)")

# ── Step 2: Copy images into genus_classifier/<Genus>/<species>/ ──────────────
print("\nStep 2: Copying into genus_classifier structure...")
copied = 0
for rec in all_records:
    dst_dir = GENUS_DIR / rec["genus"] / rec["species"]
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / rec["src_path"].name
    if not dst_path.exists():
        shutil.copy2(rec["src_path"], dst_path)
        copied += 1

print(f"  Copied {copied:,} images (already-existing files skipped)")

# ── Step 3: Build genus -> [records] map for stratified splitting ─────────────
print("\nStep 3: Building stratified train/val/test splits...")

# Group by species first (split at species level, not image level)
# This ensures all images of a species go into the same split —
# prevents data leakage from very similar intra-species photos.
species_to_records = defaultdict(list)
for rec in all_records:
    species_to_records[rec["species"]].append(rec)

# Group species by genus
genus_to_species = defaultdict(list)
for sp, recs in species_to_records.items():
    genus = recs[0]["genus"]
    genus_to_species[genus].append(sp)

train_rows, val_rows, test_rows = [], [], []

for genus, species_list in sorted(genus_to_species.items()):
    random.shuffle(species_list)
    n = len(species_list)
    n_val  = max(1, round(n * VAL_FRAC))
    n_test = max(1, round(n * TEST_FRAC))
    # Ensure at least 1 species in train even for very small genera
    n_val  = min(n_val,  n - 2) if n >= 3 else 0
    n_test = min(n_test, n - n_val - 1) if n >= 3 else 0

    val_species   = species_list[:n_val]
    test_species  = species_list[n_val:n_val + n_test]
    train_species = species_list[n_val + n_test:]

    for sp in train_species:
        for rec in species_to_records[sp]:
            train_rows.append({**rec, "split": "train",
                               "dst_path": str(GENUS_DIR / rec["genus"] / rec["species"] / rec["src_path"].name)})
    for sp in val_species:
        for rec in species_to_records[sp]:
            val_rows.append({**rec, "split": "val",
                             "dst_path": str(GENUS_DIR / rec["genus"] / rec["species"] / rec["src_path"].name)})
    for sp in test_species:
        for rec in species_to_records[sp]:
            test_rows.append({**rec, "split": "test",
                              "dst_path": str(GENUS_DIR / rec["genus"] / rec["species"] / rec["src_path"].name)})

print(f"  Train: {len(train_rows):,} images | Val: {len(val_rows):,} | Test: {len(test_rows):,}")

# ── Step 4: Write CSV files ───────────────────────────────────────────────────
CSV_FIELDS = ["split", "genus", "species", "dst_path"]

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CSV_FIELDS})

write_csv(SPLITS_DIR / "train.csv", train_rows)
write_csv(SPLITS_DIR / "val.csv",   val_rows)
write_csv(SPLITS_DIR / "test.csv",  test_rows)

all_rows = train_rows + val_rows + test_rows
write_csv(SPLITS_DIR / "all_images.csv", all_rows)

print(f"\n  Saved CSVs to {SPLITS_DIR}")

# ── Step 5: Summary ───────────────────────────────────────────────────────────
genera_list = sorted(genus_to_species.keys())
genus_counts = {g: sum(len(species_to_records[sp]) for sp in sps)
                for g, sps in genus_to_species.items()}

print("\n" + "=" * 60)
print("REORGANISATION SUMMARY")
print("=" * 60)
print(f"Total genera      : {len(genera_list)}")
print(f"Total species     : {len(species_to_records):,}")
print(f"Total images      : {len(all_records):,}")
print(f"  Train           : {len(train_rows):,} ({100*len(train_rows)/len(all_records):.1f}%)")
print(f"  Val             : {len(val_rows):,} ({100*len(val_rows)/len(all_records):.1f}%)")
print(f"  Test            : {len(test_rows):,} ({100*len(test_rows)/len(all_records):.1f}%)")
print(f"\nGenera with <20 images (flag for review):")
low_genera = [(g, c) for g, c in genus_counts.items() if c < 20]
for g, c in sorted(low_genera, key=lambda x: x[1]):
    print(f"  {g:<35} {c:>4} images")
print(f"\nTotal flagged genera: {len(low_genera)}")
print("\nDone. Next: notebook 02_baseline_resnet50.ipynb")
