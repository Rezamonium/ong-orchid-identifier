"""
Merge & Rename Dataset — ONG Orchid Identifier v2
Combines:
  1. Bird's Head dataset (E:\Claude Code\Genus Identifier\AI_Training\birdshead_genus_dataset)
     - Already in correct format: {Genus}_{Genus}_{epithet}_{4digit}.jpg
     - Merge train/ + valid/ back into single genus folders
  2. ONG scraped dataset (E:\Claude Code\scrapping ONG\photos)
     - Rename: {genus}_{epithet}_photo{nn}.jpg
         → {Genus}_{Genus}_{epithet}_{0001}.jpg
     - Skip: "Distribution of *.jpg"

Output: E:\Claude Code\ONG_v2\data\photos\{Genus}\{filename}.jpg

Generates: E:\Claude Code\ONG_v2\data\merge_report.txt
"""

import os, shutil, csv
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
BH_DIR     = Path("E:/Claude Code/Genus Identifier/AI_Training/birdshead_genus_dataset")
ONG_DIR    = Path("E:/Claude Code/scrapping ONG/photos")
OUT_DIR    = Path("E:/Claude Code/ONG_v2/data/photos")
DATA_DIR   = Path("E:/Claude Code/ONG_v2/data")

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
SKIP_GENUS = {"Other_Orchidaceae"}   # v1 catch-all class — skip for v2

OUT_DIR.mkdir(parents=True, exist_ok=True)

stats = defaultdict(lambda: {"bh": 0, "ong": 0})
log_rows = []   # for merge_report

# ════════════════════════════════════════════════════════════════════
# PART 1: Copy Bird's Head photos (merge train + valid)
# ════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Part 1: Copying Bird's Head dataset...")
print("=" * 60)

bh_total = 0
for split in ["train", "valid"]:
    split_dir = BH_DIR / split
    if not split_dir.exists():
        continue
    for genus_dir in sorted(split_dir.iterdir()):
        if not genus_dir.is_dir():
            continue
        genus = genus_dir.name
        if genus in SKIP_GENUS:
            print(f"  Skipped: {genus}")
            continue
        out_genus = OUT_DIR / genus
        out_genus.mkdir(parents=True, exist_ok=True)
        for img in genus_dir.iterdir():
            if img.is_file() and img.suffix.lower() in VALID_EXTS:
                dst = out_genus / img.name
                if not dst.exists():
                    shutil.copy2(img, dst)
                    stats[genus]["bh"] += 1
                    bh_total += 1
                # if exists: same filename from train/valid — skip duplicate

print(f"  Bird's Head photos copied: {bh_total:,}")

# ════════════════════════════════════════════════════════════════════
# PART 2: Rename & copy ONG scraped photos
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 2: Renaming & copying ONG scraped photos...")
print("=" * 60)

ong_total  = 0
ong_skip   = 0

for sp_dir in sorted(ONG_DIR.iterdir()):
    if not sp_dir.is_dir():
        continue

    species_name = sp_dir.name                         # e.g. "Bulbophyllum aechmophorum"
    genus        = species_name.split()[0]             # e.g. "Bulbophyllum"

    # Fix known typo
    if genus == "AgrostophylIum":
        genus = "Agrostophyllum"

    # Species name → filename-safe (spaces → underscores, preserve dots)
    sp_underscore = species_name.replace(" ", "_")     # e.g. "Bulbophyllum_aechmophorum"

    # Collect valid orchid photos (skip distribution maps)
    imgs = sorted([
        f for f in sp_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in VALID_EXTS
        and not f.name.startswith("Distribution of")
    ])

    if not imgs:
        ong_skip += 1
        continue

    out_genus = OUT_DIR / genus
    out_genus.mkdir(parents=True, exist_ok=True)

    for seq, img in enumerate(imgs, start=5001):
        # ONG photos start at 5001 to avoid collision with BH numbers (max ~2000)
        # Format: {Genus}_{Species}_{seq:04d}.jpg
        new_name = f"{genus}_{sp_underscore}_{seq:04d}.jpg"
        dst      = out_genus / new_name
        if not dst.exists():
            shutil.copy2(img, dst)
            stats[genus]["ong"] += 1
            ong_total += 1
        # else: collision — should not happen with 5001+ offset

    log_rows.append({
        "species":    species_name,
        "genus":      genus,
        "ong_photos": len(imgs),
    })

print(f"  ONG photos renamed & copied: {ong_total:,}")
print(f"  Species skipped (0 photos):  {ong_skip}")

# ════════════════════════════════════════════════════════════════════
# REPORT
# ════════════════════════════════════════════════════════════════════
total_photos = sum(d["bh"] + d["ong"] for d in stats.values())
genera_list  = sorted(stats.keys())

report_lines = [
    "=" * 60,
    "MERGED DATASET REPORT",
    "=" * 60,
    f"Bird's Head photos  : {bh_total:,}",
    f"ONG scraped photos  : {ong_total:,}",
    f"Total photos        : {total_photos:,}",
    f"Total genera        : {len(genera_list)}",
    "",
    f"{'Genus':<35} {'BH':>6} {'ONG':>6} {'Total':>7}",
    "-" * 55,
]
for g in genera_list:
    bh  = stats[g]["bh"]
    ong = stats[g]["ong"]
    report_lines.append(f"{g:<35} {bh:>6} {ong:>6} {bh+ong:>7}")

report_lines += [
    "",
    f"Genera with 0 BH photos  : {sum(1 for d in stats.values() if d['bh']==0)}",
    f"Genera with 0 ONG photos : {sum(1 for d in stats.values() if d['ong']==0)}",
    f"Genera in both datasets  : {sum(1 for d in stats.values() if d['bh']>0 and d['ong']>0)}",
]

report_text = "\n".join(report_lines)
print("\n" + report_text)

with open(DATA_DIR / "merge_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

# Save per-genus counts CSV
with open(DATA_DIR / "merged_genus_counts.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["genus", "bh_photos", "ong_photos", "total"])
    for g in genera_list:
        w.writerow([g, stats[g]["bh"], stats[g]["ong"], stats[g]["bh"]+stats[g]["ong"]])

print(f"\nSaved: {DATA_DIR / 'merge_report.txt'}")
print(f"Saved: {DATA_DIR / 'merged_genus_counts.csv'}")
print(f"\nDone. Photos at: {OUT_DIR}")
