"""
Dataset Audit Script — ONG Orchid Identifier v2
Day 1-2: Quality check on E:\Claude Code\scrapping ONG\photos

Checks:
  - Image counts per species
  - Empty folders
  - Corrupt / unreadable images
  - Non-standard file extensions
  - Genus distribution
  - Generates: audit_report.txt, species_counts.csv, genus_counts.csv
"""

import os
import csv
import json
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("WARNING: Pillow not installed — corrupt image check skipped.")

# ── Config ──────────────────────────────────────────────────────────────────
PHOTOS_DIR   = Path("E:/Claude Code/scrapping ONG/photos")
OUTPUT_DIR   = Path("E:/Claude Code/ONG_v2/data")
VALID_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
# ────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Scanning: {PHOTOS_DIR}")
print("=" * 60)

species_rows   = []   # (species, genus, img_count, corrupt, bad_ext)
genus_counts   = defaultdict(lambda: {"species": 0, "images": 0})
corrupt_list   = []
bad_ext_list   = []
empty_species  = []

species_dirs = sorted([d for d in PHOTOS_DIR.iterdir() if d.is_dir()])
total_species = len(species_dirs)

for i, sp_dir in enumerate(species_dirs, 1):
    species_name = sp_dir.name
    # Genus = first word of species name
    genus = species_name.split()[0] if species_name else "Unknown"

    all_files  = list(sp_dir.iterdir())
    img_files  = [f for f in all_files
                  if f.is_file()
                  and f.suffix.lower() in VALID_EXTS
                  and not f.name.startswith("Distribution of")]
    bad_files  = [f for f in all_files if f.is_file() and f.suffix.lower() not in VALID_EXTS]

    corrupt_count  = 0
    if PIL_AVAILABLE:
        for img_path in img_files:
            try:
                with Image.open(img_path) as im:
                    im.verify()
            except Exception:
                corrupt_count += 1
                corrupt_list.append(str(img_path))

    if bad_files:
        bad_ext_list.extend([(str(f), f.suffix) for f in bad_files])

    img_count = len(img_files)

    if img_count == 0:
        empty_species.append(species_name)

    genus_counts[genus]["species"] += 1
    genus_counts[genus]["images"]  += img_count

    species_rows.append({
        "species":        species_name,
        "genus":          genus,
        "img_count":      img_count,
        "corrupt":        corrupt_count,
        "bad_ext_files":  len(bad_files),
    })

    if i % 200 == 0:
        print(f"  [{i}/{total_species}] processed...")

# ── Write species_counts.csv ─────────────────────────────────────────────────
species_csv = OUTPUT_DIR / "species_counts.csv"
with open(species_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["species","genus","img_count","corrupt","bad_ext_files"])
    writer.writeheader()
    writer.writerows(species_rows)
print(f"\nSaved: {species_csv}")

# ── Write genus_counts.csv ───────────────────────────────────────────────────
genus_csv = OUTPUT_DIR / "genus_counts.csv"
with open(genus_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["genus","species_count","image_count","avg_imgs_per_species"])
    for genus, d in sorted(genus_counts.items()):
        avg = d["images"] / d["species"] if d["species"] > 0 else 0
        writer.writerow([genus, d["species"], d["images"], round(avg, 1)])
print(f"Saved: {genus_csv}")

# ── Summary report ───────────────────────────────────────────────────────────
total_images  = sum(r["img_count"] for r in species_rows)
total_corrupt = sum(r["corrupt"]   for r in species_rows)
total_bad_ext = sum(r["bad_ext_files"] for r in species_rows)
total_genera  = len(genus_counts)
genera_50plus = sum(1 for d in genus_counts.values() if d["images"] >= 50)
genera_20plus = sum(1 for d in genus_counts.values() if d["images"] >= 20)

counts = [r["img_count"] for r in species_rows]
counts_sorted = sorted(counts)
median_imgs = counts_sorted[len(counts_sorted)//2]
avg_imgs    = total_images / total_species if total_species else 0
max_imgs    = max(counts) if counts else 0
min_imgs    = min(counts) if counts else 0

report_lines = [
    "=" * 60,
    "ONG v2 DATASET AUDIT REPORT",
    "=" * 60,
    f"Photos directory : {PHOTOS_DIR}",
    f"",
    f"--- OVERALL ---",
    f"Total species    : {total_species:,}",
    f"Total images     : {total_images:,}",
    f"Total genera     : {total_genera}",
    f"",
    f"--- IMAGE COUNTS PER SPECIES ---",
    f"Average          : {avg_imgs:.1f}",
    f"Median           : {median_imgs}",
    f"Min              : {min_imgs}",
    f"Max              : {max_imgs}",
    f"Empty (0 images) : {len(empty_species)}",
    f"",
    f"--- GENUS COVERAGE ---",
    f"Genera with 50+ images  : {genera_50plus}",
    f"Genera with 20+ images  : {genera_20plus}",
    f"",
    f"--- DATA QUALITY ---",
    f"Corrupt images   : {total_corrupt}",
    f"Non-image files  : {total_bad_ext}",
    f"",
    f"--- EMPTY SPECIES FOLDERS ({len(empty_species)}) ---",
]
for sp in empty_species:
    report_lines.append(f"  {sp}")

report_lines += [
    "",
    "--- TOP 20 GENERA BY IMAGE COUNT ---",
    f"{'Genus':<35} {'Species':>8} {'Images':>8} {'Avg':>6}",
    "-" * 60,
]
top20 = sorted(genus_counts.items(), key=lambda x: -x[1]["images"])[:20]
for genus, d in top20:
    avg = d["images"] / d["species"] if d["species"] else 0
    report_lines.append(f"{genus:<35} {d['species']:>8} {d['images']:>8} {avg:>6.1f}")

report_lines += [
    "",
    "--- BOTTOM 10 GENERA BY IMAGE COUNT ---",
    f"{'Genus':<35} {'Species':>8} {'Images':>8}",
    "-" * 60,
]
bottom10 = sorted(genus_counts.items(), key=lambda x: x[1]["images"])[:10]
for genus, d in bottom10:
    report_lines.append(f"{genus:<35} {d['species']:>8} {d['images']:>8}")

report_text = "\n".join(report_lines)
print("\n" + report_text)

report_path = OUTPUT_DIR / "audit_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\nSaved: {report_path}")
print("\nAudit complete.")
