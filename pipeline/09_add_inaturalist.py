"""
Add iNaturalist Research-Grade Photos — ONG Orchid Identifier v2

Source: E:/Claude Code/iNaturalist scrapping/inaturalist_orchids/
Structure: {Genus}_{epithet}/ -> photos already named:
  {Genus}_{Genus}_{epithet}_{obs_id}_{photo_idx}_{username}_inaturalist.jpg

Output name (add _RG suffix):
  {Genus}_{Genus}_{epithet}_{obs_id}_{photo_idx}_{username}_inaturalist_RG.jpg

Steps:
  1. Copy photos to data/photos/{Genus}/ with _RG suffix
  2. Add to combined species map (bh_species_map.json)
  3. Append to image_type_labels.csv as 'live'
  4. Regenerate splits
"""

import shutil, csv, json
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
INAT_SRC   = Path("E:/Claude Code/iNaturalist scrapping/inaturalist_orchids")
OUT_DIR    = Path("E:/Claude Code/ONG_v2/data/photos")
DATA_DIR   = Path("E:/Claude Code/ONG_v2/data")
SPLITS_DIR = DATA_DIR / "splits"
SP_MAP     = DATA_DIR / "bh_species_map.json"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ════════════════════════════════════════════════════════════════════
# STEP 1: Copy photos with _RG suffix
# ════════════════════════════════════════════════════════════════════
print("Step 1: Copying iNaturalist photos (adding _RG suffix)...")

stats       = defaultdict(int)
new_sp_map  = {}   # new_filename -> species
log_rows    = []
inat_total  = 0
skipped_dup = 0

for sp_dir in sorted(INAT_SRC.iterdir()):
    if not sp_dir.is_dir():
        continue

    # Folder name: {Genus}_{epithet}  (e.g. "Bulbophyllum_antenniferum")
    parts       = sp_dir.name.split("_")
    genus       = parts[0]
    epithet     = " ".join(parts[1:])
    species_str = f"{genus} {epithet}"

    out_genus = OUT_DIR / genus
    out_genus.mkdir(parents=True, exist_ok=True)

    imgs = sorted([
        f for f in sp_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in VALID_EXTS
        and not f.name.startswith("._")
        and not f.name.startswith(".")
    ])

    if not imgs:
        continue

    for img in imgs:
        new_name = img.stem + "_RG" + img.suffix
        dst = out_genus / new_name
        if not dst.exists():
            shutil.copy2(img, dst)
            stats[genus] += 1
            inat_total += 1
        else:
            skipped_dup += 1
        new_sp_map[new_name] = species_str

    log_rows.append({
        "genus":   genus,
        "species": species_str,
        "photos":  len(imgs),
    })

print(f"  Copied  : {inat_total:,} photos")
print(f"  Skipped : {skipped_dup:,} (already exist)")
print(f"  Genera  : {len(stats)}")

# ════════════════════════════════════════════════════════════════════
# STEP 2: Update combined species map
# ════════════════════════════════════════════════════════════════════
existing_map = {}
if SP_MAP.exists():
    with open(SP_MAP, encoding="utf-8") as f:
        existing_map = json.load(f)

combined = {**existing_map, **new_sp_map}
with open(SP_MAP, "w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)
print(f"\nStep 2: Species map updated ({len(combined):,} entries)")

# ════════════════════════════════════════════════════════════════════
# STEP 3: Append to image_type_labels.csv
# ════════════════════════════════════════════════════════════════════
print("\nStep 3: Updating image_type_labels.csv...")

labels_path = DATA_DIR / "image_type_labels.csv"
existing_paths = set()
fieldnames = None
with open(labels_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        existing_paths.add(row["path"])

new_entries = []
for genus_dir in sorted(OUT_DIR.iterdir()):
    if not genus_dir.is_dir():
        continue
    genus = genus_dir.name
    for f in sorted(genus_dir.iterdir()):
        if not (f.is_file() and f.suffix.lower() in VALID_EXTS):
            continue
        if not f.name.endswith("_RG.jpg"):
            continue
        path_str = str(f)
        if path_str in existing_paths:
            continue
        new_entries.append({
            "path":         path_str,
            "genus":        genus,
            "filename":     f.name,
            "label":        "live",
            "white_pct":    "",
            "light_pct":    "",
            "mean_sat":     "",
            "high_sat_pct": "",
        })

with open(labels_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writerows(new_entries)

print(f"  Added {len(new_entries):,} iNaturalist entries as 'live'")

# ════════════════════════════════════════════════════════════════════
# STEP 4: Regenerate splits
# ════════════════════════════════════════════════════════════════════
print("\nStep 4: Regenerating splits...")

import random

TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

with open(SP_MAP, encoding="utf-8") as f:
    sp_lookup = json.load(f)

def parse_species(filename: str) -> str:
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

print(f"\n  Total : {len(all_records):,} images | {len(set(r['genus'] for r in all_records))} genera | {len(sp_map):,} species")
print(f"  Train : {len(train_rows):,} | Val: {len(val_rows):,} | Test: {len(test_rows):,}")

# ── Write report ──────────────────────────────────────────────────────────────
report_path = DATA_DIR / "inaturalist_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("iNaturalist Research-Grade Photos — Integration Report\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Source  : {INAT_SRC}\n")
    f.write(f"Folders : {len(log_rows)}\n")
    f.write(f"Photos  : {inat_total:,}\n\n")
    f.write(f"{'Genus':<30} {'Photos':>8}\n")
    f.write("-" * 40 + "\n")
    for genus in sorted(stats.keys()):
        f.write(f"{genus:<30} {stats[genus]:>8}\n")
    f.write("\nSpecies detail:\n")
    f.write("-" * 60 + "\n")
    for row in log_rows:
        f.write(f"  {row['species']:<50} {row['photos']:>3} photos\n")

print(f"\nDone! Report: {report_path}")
