"""
Add Jeffrey Champion Photos — ONG Orchid Identifier v2

Source: C:/Users/rezas/Desktop/1. Database Orchids of New Guinea/4. Orchid Photos Jeffrey Champion
Structure:
  - Most genera: {Genus}/{Species}/{photos}
  - 7 sectioned genera: {Genus}/{Section}/{Species}/{photos}
    (Bulbophyllum, Coelogyne, Dendrobium, Dendrochillum, Glomera, Thrixspermum, XXXFamily Eria)

Filter criteria:
  1. Identified species: folder name matches NNG species list (orchidsnewguinea_data_clean.xlsx)
  2. Unidentified specimens: folder name contains "Papua" (labelled at genus level)

Exclude:
  - macOS hidden files (._* and .DS_Store)
  - Identified species NOT in NNG list
  - "Papua" folders (SP unidentified) already handled — included with genus label

Rename to: {Genus}_{Species}_{seq:04d}.jpg  (starting at 9001 per species)
Copy to:   E:/Claude Code/ONG_v2/data/photos/{Genus}/

Output: data/jeffrey_champion_report.txt
"""

import shutil, csv
import openpyxl
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
JC_DIR   = Path(r"C:\Users\rezas\Desktop\1. Database Orchids of New Guinea\4. Orchid Photos Jeffrey Champion")
OUT_DIR  = Path("E:/Claude Code/ONG_v2/data/photos")
DATA_DIR = Path("E:/Claude Code/ONG_v2/data")

VALID_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
SECTIONED     = {"Bulbophyllum", "Coelogyne", "Dendrobium", "Dendrochillum",
                 "Glomera", "Thrixspermum", "XXXFamily Eria"}
SEQ_START     = 9001   # Jeffrey Champion sequence start (avoid collision with BH 1-4999, ONG 5001-8999)

# ── Load NNG species list ─────────────────────────────────────────────────────
print("Loading NNG species list...")
wb = openpyxl.load_workbook(DATA_DIR / "orchidsnewguinea_data_clean.xlsx", read_only=True)
ws = wb.active
nng_species = set()
for row in ws.iter_rows(min_row=2, values_only=True):
    if row[0]:
        nng_species.add(row[0].strip().lower())
wb.close()
print(f"  {len(nng_species):,} species in NNG list")

# ── Helper: get leaf (species) folders under a genus dir ─────────────────────
def get_species_folders(genus_dir, genus_name):
    """
    Returns list of (species_folder_path, species_label) tuples.
    Handles both direct (genus/species) and sectioned (genus/section/species) structures.
    """
    results = []
    # Check if direct or sectioned by testing if first subdir contains images
    subdirs = [d for d in genus_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not subdirs:
        return results

    # Detect structure: if first subdir has image files → direct; if only subdirs → sectioned
    first_sub_items = list(subdirs[0].iterdir())
    has_imgs_directly = any(
        f.is_file() and f.suffix.lower() in VALID_EXTS and not f.name.startswith('._')
        for f in first_sub_items
    )

    if genus_name in SECTIONED or (not has_imgs_directly and any(d.is_dir() for d in first_sub_items)):
        # Sectioned: genus/section/species/photos
        for section_dir in subdirs:
            for sp_dir in section_dir.iterdir():
                if sp_dir.is_dir() and not sp_dir.name.startswith('.'):
                    results.append(sp_dir)
    else:
        # Direct: genus/species/photos
        for sp_dir in subdirs:
            results.append(sp_dir)
    return results

# ── Process ───────────────────────────────────────────────────────────────────
stats        = defaultdict(lambda: {"copied": 0, "skipped_species": 0, "skipped_files": 0})
total_copied = 0
total_skipped_sp = 0
total_skipped_files = 0
log_rows     = []

print(f"\nProcessing Jeffrey Champion photos...")
print("=" * 60)

for genus_dir in sorted(JC_DIR.iterdir()):
    if not genus_dir.is_dir():
        continue
    genus = genus_dir.name
    if genus.startswith('.') or genus == "XXXFamily Eria":
        continue   # skip macOS hidden dirs and non-genus folders

    out_genus = OUT_DIR / genus
    out_genus.mkdir(parents=True, exist_ok=True)

    species_folders = get_species_folders(genus_dir, genus)

    for sp_dir in species_folders:
        sp_folder_name = sp_dir.name   # e.g. "Bulbophyllum elongatum" or "Dendrobium SP 2 Papua"

        # ── Determine if this species should be included ──────────────────────
        sp_lower = sp_folder_name.lower()
        is_papua = "papua" in sp_lower
        is_in_nng = sp_lower in nng_species

        if not is_in_nng and not is_papua:
            stats[genus]["skipped_species"] += 1
            total_skipped_sp += 1
            continue   # not NNG species and not Papua specimen

        # ── Build species label for filename ──────────────────────────────────
        # Use folder name, replace spaces with underscores, remove problematic chars
        sp_label = sp_folder_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        # Remove any chars that aren't alphanumeric, underscore, or hyphen
        sp_label = "".join(c for c in sp_label if c.isalnum() or c in "_-")

        # Collect image files (skip macOS hidden files)
        imgs = sorted([
            f for f in sp_dir.iterdir()
            if f.is_file()
            and f.suffix.lower() in VALID_EXTS
            and not f.name.startswith("._")
            and not f.name.startswith(".")
        ])

        if not imgs:
            continue

        # ── Find the next available sequence number for this species ──────────
        # Check existing files to avoid collision
        existing = list(out_genus.glob(f"{genus}_{sp_label}_*.jpg"))
        if existing:
            # Find highest existing seq for this species
            max_seq = SEQ_START - 1
            for ef in existing:
                try:
                    seq_part = ef.stem.split("_")[-1]
                    seq_num = int(seq_part)
                    if seq_num > max_seq:
                        max_seq = seq_num
                except ValueError:
                    pass
            seq_start = max(max_seq + 1, SEQ_START)
        else:
            seq_start = SEQ_START

        # ── Copy and rename ───────────────────────────────────────────────────
        for seq, img in enumerate(imgs, start=seq_start):
            new_name = f"{genus}_{sp_label}_{seq:04d}.jpg"
            dst = out_genus / new_name
            if not dst.exists():
                shutil.copy2(img, dst)
                stats[genus]["copied"] += 1
                total_copied += 1
            else:
                stats[genus]["skipped_files"] += 1
                total_skipped_files += 1

        log_rows.append({
            "genus": genus,
            "species_folder": sp_folder_name,
            "sp_label": sp_label,
            "is_nng": is_in_nng,
            "is_papua": is_papua,
            "photos_copied": len(imgs),
        })

    print(f"  {genus}: {stats[genus]['copied']:>4} copied, {stats[genus]['skipped_species']:>3} species skipped")

# ── Write report ──────────────────────────────────────────────────────────────
report_path = DATA_DIR / "jeffrey_champion_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("Jeffrey Champion Photos — Integration Report\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total photos copied : {total_copied:,}\n")
    f.write(f"Species skipped     : {total_skipped_sp:,} (not in NNG list, not Papua)\n")
    f.write(f"Files skipped (dup) : {total_skipped_files:,}\n\n")
    f.write(f"{'Genus':<30} {'Copied':>8} {'Sp.Skipped':>12}\n")
    f.write("-" * 54 + "\n")
    for genus in sorted(stats.keys()):
        f.write(f"{genus:<30} {stats[genus]['copied']:>8} {stats[genus]['skipped_species']:>12}\n")
    f.write("\n\nIncluded species detail:\n")
    f.write("-" * 60 + "\n")
    for row in log_rows:
        tag = "[NNG]" if row["is_nng"] else "[PAPUA]"
        f.write(f"  {tag} {row['genus']} / {row['species_folder']}  ({row['photos_copied']} photos)\n")

print("\n" + "=" * 60)
print(f"DONE")
print(f"  Photos copied : {total_copied:,}")
print(f"  Species skipped (not NNG/Papua): {total_skipped_sp:,}")
print(f"  Files skipped (already exist)  : {total_skipped_files:,}")
print(f"  Report: {report_path}")
print("\nNext steps:")
print("  1. Run: python notebooks/05_detect_image_type.py  (re-run to include JC photos)")
print("  2. Run: python notebooks/04_generate_splits.py    (re-generate splits)")
print("  3. Re-zip data/photos/ for Colab upload")
