"""
Replace Bird's Head Photos from Lucid Key — ONG Orchid Identifier v2

Source: F:/00. Lucid Birdshead Orchids/Birdshead Orchid v1.2/key/Media/Html/assets/image/
Structure: {genus}_{epithet}/ -> photos (skip thumbs/ subfolder)

Naming: keep original filename, only change the prefix:
  ania_penangiana_img_6755_rs.jpg  ->  Ania_Ania_penangiana_img_6755_rs.jpg

Species mapping saved to data/bh_species_map.json for use by parse_species.

Steps:
  1. Delete existing BH photos from data/photos/ (no attribution suffix = BH)
  2. Copy & rename from Lucid source (original filename, capitalised prefix)
  3. Save species mapping JSON
  4. Update image_type_labels.csv
  5. Regenerate splits

Output: data/bh_lucid_report.txt
"""

import os, shutil, csv, json
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
LUCID_SRC  = Path(r"F:\00. Lucid Birdshead Orchids\Birdshead Orchid v1.2\key\Media\Html\assets\image")
OUT_DIR    = Path("E:/Claude Code/ONG_v2/data/photos")
DATA_DIR   = Path("E:/Claude Code/ONG_v2/data")
SPLITS_DIR = DATA_DIR / "splits"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ════════════════════════════════════════════════════════════════════
# STEP 1: Delete existing BH photos (no attribution suffix in stem)
# ════════════════════════════════════════════════════════════════════
print("Step 1: Removing existing BH photos (no attribution suffix)...")
deleted = 0
for genus_dir in OUT_DIR.iterdir():
    if not genus_dir.is_dir():
        continue
    for f in list(genus_dir.iterdir()):
        if not (f.is_file() and f.suffix.lower() in VALID_EXTS):
            continue
        fn = f.name
        # BH files have neither ONG nor JC attribution
        if "Orchids of New Guinea" not in fn and "Jeffrey Champion" not in fn:
            f.unlink()
            deleted += 1

print(f"  Deleted {deleted:,} BH photos")

# ════════════════════════════════════════════════════════════════════
# STEP 2: Copy & rename from Lucid source
# ════════════════════════════════════════════════════════════════════
print("\nStep 2: Copying from Lucid source (original filenames)...")

stats        = defaultdict(int)
species_map  = {}   # new_filename -> species name  (for parse_species)
log_rows     = []
bh_total     = 0

for sp_dir in sorted(LUCID_SRC.iterdir()):
    if not sp_dir.is_dir():
        continue

    folder_name = sp_dir.name           # e.g. "ania_penangiana"
    parts       = folder_name.split("_")
    if len(parts) < 2:
        print(f"  Skipping unrecognised folder: {folder_name}")
        continue

    genus_lower  = parts[0]             # e.g. "ania"
    epithet_part = "_".join(parts[1:])  # e.g. "penangiana" or "var_papuanum"
    genus        = genus_lower.capitalize()   # e.g. "Ania"
    species_name = f"{genus} {epithet_part.replace('_', ' ')}"  # e.g. "Ania penangiana"

    out_genus = OUT_DIR / genus
    out_genus.mkdir(parents=True, exist_ok=True)

    # Collect images (skip thumbs subfolder and hidden files)
    imgs = sorted([
        f for f in sp_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in VALID_EXTS
        and not f.name.startswith(".")
        and not f.name.startswith("._")
    ])

    if not imgs:
        continue

    for img in imgs:
        orig_stem = img.stem   # e.g. "ania_penangiana_img_6755_rs"

        # Strip lowercase {genus}_{epithet}_ prefix if present
        prefix = folder_name + "_"
        if orig_stem.lower().startswith(prefix):
            suffix_part = orig_stem[len(prefix):]
        else:
            suffix_part = orig_stem   # no recognisable prefix (e.g. "351")

        new_stem = f"{genus}_{genus}_{epithet_part}_{suffix_part}"
        new_name = new_stem + ".jpg"
        dst = out_genus / new_name

        if not dst.exists():
            shutil.copy2(img, dst)
            stats[genus] += 1
            bh_total += 1
            species_map[new_name] = species_name

    log_rows.append({
        "genus":   genus,
        "species": species_name,
        "photos":  len(imgs),
    })

print(f"  Copied {bh_total:,} BH photos from {len(log_rows)} species")

# ════════════════════════════════════════════════════════════════════
# STEP 3: Save species mapping JSON
# ════════════════════════════════════════════════════════════════════
map_path = DATA_DIR / "bh_species_map.json"
with open(map_path, "w", encoding="utf-8") as f:
    json.dump(species_map, f, ensure_ascii=False, indent=2)
print(f"\nStep 3: Saved species map ({len(species_map):,} entries) -> {map_path}")

# ════════════════════════════════════════════════════════════════════
# STEP 4: Update image_type_labels.csv
# ════════════════════════════════════════════════════════════════════
print("\nStep 4: Updating image_type_labels.csv...")

labels_path = DATA_DIR / "image_type_labels.csv"

# Remove old BH entries (no attribution suffix in filename)
kept_rows      = []
removed_labels = 0
with open(labels_path, encoding="utf-8") as f:
    reader     = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        fn = row["filename"]
        if "Orchids of New Guinea" not in fn and "Jeffrey Champion" not in fn:
            removed_labels += 1
            continue
        kept_rows.append(row)

print(f"  Removed {removed_labels:,} old BH entries")

# Add new BH entries — all files without known attribution suffix
existing_paths = {row["path"] for row in kept_rows}
new_entries = []
for genus_dir in sorted(OUT_DIR.iterdir()):
    if not genus_dir.is_dir():
        continue
    genus = genus_dir.name
    for f in sorted(genus_dir.iterdir()):
        if not (f.is_file() and f.suffix.lower() in VALID_EXTS):
            continue
        fn = f.name
        path_str = str(f)
        if path_str in existing_paths:
            continue   # already in kept_rows (ONG or JC)
        # Must be a BH file (no attribution suffix, new from this run)
        new_entries.append({
            "path":         path_str,
            "genus":        genus,
            "filename":     fn,
            "label":        "live",
            "white_pct":    "",
            "light_pct":    "",
            "mean_sat":     "",
            "high_sat_pct": "",
        })

print(f"  Adding {len(new_entries):,} new BH entries as 'live'")

with open(labels_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(kept_rows)
    writer.writerows(new_entries)

print(f"  Total entries: {len(kept_rows) + len(new_entries):,}")

# ════════════════════════════════════════════════════════════════════
# STEP 5: Regenerate splits
# ════════════════════════════════════════════════════════════════════
print("\nStep 5: Regenerating splits...")

import random

TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Load BH species map
with open(map_path, encoding="utf-8") as f:
    bh_sp_map = json.load(f)

def parse_species(filename: str) -> str:
    """
    Extract species name.
    - BH: look up in bh_species_map (original camera-code filenames)
    - ONG/JC: strip attribution suffix, then strip seq number
    """
    # BH lookup
    if filename in bh_sp_map:
        return bh_sp_map[filename]

    # ONG / JC: {Genus}_{Genus}_{epithet}_{seq}[_Attribution]
    stem  = Path(filename).stem
    parts = stem.split("_")
    while parts and not parts[-1].isdigit():
        parts = parts[:-1]
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    species_parts = parts[1:]
    return " ".join(species_parts)

all_records = []
for genus_dir in sorted(OUT_DIR.iterdir()):
    if not genus_dir.is_dir():
        continue
    genus = genus_dir.name
    for img in sorted(genus_dir.iterdir()):
        if img.is_file() and img.suffix.lower() in VALID_EXTS:
            all_records.append({
                "path":    str(img),
                "genus":   genus,
                "species": parse_species(img.name),
            })

sp_map    = defaultdict(list)
for rec in all_records:
    sp_map[rec["species"]].append(rec)

genus_sp = defaultdict(list)
for sp, recs in sp_map.items():
    genus_sp[recs[0]["genus"]].append(sp)

train_rows, val_rows, test_rows = [], [], []
for genus, species_list in sorted(genus_sp.items()):
    random.shuffle(species_list)
    n  = len(species_list)
    nv = max(1, round(n * VAL_FRAC))  if n >= 3 else 0
    nt = max(1, round(n * VAL_FRAC))  if n >= 3 else 0
    nv = min(nv, n - 2)               if n >= 3 else 0
    nt = min(nt, n - nv - 1)          if n >= 3 else 0
    for sp in species_list[:nv]:
        for r in sp_map[sp]: val_rows.append({**r, "split": "val"})
    for sp in species_list[nv:nv+nt]:
        for r in sp_map[sp]: test_rows.append({**r, "split": "test"})
    for sp in species_list[nv+nt:]:
        for r in sp_map[sp]: train_rows.append({**r, "split": "train"})

FIELDS = ["split", "genus", "species", "path"]
def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows: w.writerow({k: row[k] for k in FIELDS})

save_csv(SPLITS_DIR / "train.csv",      train_rows)
save_csv(SPLITS_DIR / "val.csv",        val_rows)
save_csv(SPLITS_DIR / "test.csv",       test_rows)
save_csv(SPLITS_DIR / "all_images.csv", train_rows + val_rows + test_rows)

print(f"  Total : {len(all_records):,} images | {len(set(r['genus'] for r in all_records))} genera | {len(sp_map):,} species")
print(f"  Train : {len(train_rows):,}")
print(f"  Val   : {len(val_rows):,}")
print(f"  Test  : {len(test_rows):,}")

# Live splits
live_set = set()
with open(labels_path, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["label"] in ("live", "uncertain"):
            live_set.add(row["path"])

for split_name in ["train", "val", "test"]:
    src = SPLITS_DIR / f"{split_name}.csv"
    dst = SPLITS_DIR / f"{split_name}_live.csv"
    kept, total = 0, 0
    with open(src, encoding="utf-8") as fin, \
         open(dst, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            total += 1
            if row["path"] in live_set:
                writer.writerow(row)
                kept += 1
    print(f"  {split_name}_live: {kept:,} / {total:,}")

# ── Write report ──────────────────────────────────────────────────────────────
report_path = DATA_DIR / "bh_lucid_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("Bird's Head (Lucid Key) — Integration Report\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Source : {LUCID_SRC}\n")
    f.write(f"Species: {len(log_rows)}\n")
    f.write(f"Photos : {bh_total:,}\n\n")
    f.write(f"{'Genus':<30} {'Photos':>8}\n")
    f.write("-" * 40 + "\n")
    for genus in sorted(stats.keys()):
        f.write(f"{genus:<30} {stats[genus]:>8}\n")
    f.write("\nSpecies detail:\n")
    f.write("-" * 60 + "\n")
    for row in log_rows:
        f.write(f"  {row['species']:<50} {row['photos']:>4} photos\n")

print(f"\nDone! Report: {report_path}")
