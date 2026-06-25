"""
Generate Train/Val/Test Splits — ONG Orchid Identifier v3

Scans the (already manually curated) live-only dataset at data/photos/{Genus}/.
Illustrations & herbarium were physically removed in Phase 0 (moved to
E:\\Claude Code\\Botanical Illustration), so the photo folder is the source of
truth and NO automatic image-type detector is re-run here (that detector had
false positives and would discard good live photos after manual curation).

Photos use format: {Genus}_{Genus}_{epithet}_{seq:04d}[_Attribution].jpg
Species is parsed from the filename for stratified splitting.

Stratification: split at SPECIES level (all photos of a species → same split)
to avoid leakage. Improvements over v2:
  - Portable paths (derived from this file's location → works in v3).
  - Corrupt/unreadable images are verified and skipped (replaces the old
    05_detect_image_type.py "error" exclusion).
  - Better genus coverage in val/test:
        n_species == 1 → train only (cannot split without leakage)
        n_species == 2 → 1 train + 1 val
        n_species >= 3 → >=1 val AND >=1 test guaranteed
  - Writes both base splits AND *_live.csv (identical post-curation) so the
    training script (03) and FAISS builder (04_build) work unchanged.

Output (data/splits/): train.csv, val.csv, test.csv, all_images.csv,
                       train_live.csv, val_live.csv, test_live.csv,
                       unreadable_images.txt, split_coverage_report.txt
"""

import csv, json, random, hashlib
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Install Pillow first: pip install Pillow")

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent        # E:/Claude Code/ONG_v3
PHOTOS_DIR   = PROJECT_ROOT / "data" / "photos"
SPLITS_DIR   = PROJECT_ROOT / "data" / "splits"
BH_MAP_PATH  = PROJECT_ROOT / "data" / "bh_species_map.json"
VALID_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15
RANDOM_SEED  = 42
VERIFY_IMAGES = True   # set False to skip the (slower) corruption check

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
random.seed(RANDOM_SEED)

# Load BH species map (original-filename -> species name)
BH_SPECIES_MAP = {}
if BH_MAP_PATH.exists():
    with open(BH_MAP_PATH, encoding="utf-8") as f:
        BH_SPECIES_MAP = json.load(f)


def parse_species(filename: str) -> str:
    """Extract 'Genus epithet [var. x]' from filename. BH files use the map."""
    if filename in BH_SPECIES_MAP:
        return BH_SPECIES_MAP[filename]
    stem  = Path(filename).stem
    parts = stem.split("_")
    while parts and not parts[-1].isdigit():
        parts = parts[:-1]
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts[1:])


def is_readable(p: Path) -> bool:
    if not VERIFY_IMAGES:
        return True
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


# ── Collect + verify all images ───────────────────────────────────────────────
print(f"Scanning {PHOTOS_DIR} ...")
all_records, unreadable = [], []
genus_dirs = [d for d in sorted(PHOTOS_DIR.iterdir()) if d.is_dir()]
for gi, genus_dir in enumerate(genus_dirs, 1):
    genus = genus_dir.name
    for img in sorted(genus_dir.iterdir()):
        if not (img.is_file() and img.suffix.lower() in VALID_EXTS):
            continue
        if not is_readable(img):
            unreadable.append(str(img))
            continue
        all_records.append({"path": str(img), "genus": genus,
                            "species": parse_species(img.name)})
    if gi % 20 == 0:
        print(f"  [{gi}/{len(genus_dirs)}] genera scanned...")

n_genera  = len(set(r["genus"] for r in all_records))
n_species = len(set(r["species"] for r in all_records))
print(f"\n  Readable images: {len(all_records):,}")
print(f"  Unreadable     : {len(unreadable):,} (skipped)")
print(f"  Genera         : {n_genera}")
print(f"  Species        : {n_species:,}")

# ── Curation-leak guard ───────────────────────────────────────────────────────
# v2's training set was filtered by the user's manual curation in
# image_type_labels_2ndrev.csv (label 'notused' = rejected). The v3 migration once
# dropped that filter, silently re-including 6,907 rejected images. Warn loudly if any
# 'notused' file is still present so it cannot recur unnoticed.
LABELS_2NDREV = Path(r"E:\Claude Code\ONG_v2\data\image_type_labels_2ndrev.csv")
if LABELS_2NDREV.exists():
    notused_keys = set()
    with open(LABELS_2NDREV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("label") == "notused":
                notused_keys.add(f'{row["genus"]}||{row["filename"]}')
    leaked = [r for r in all_records
              if f'{r["genus"]}||{Path(r["path"]).name}' in notused_keys]
    if leaked:
        print(f"\n  !! CURATION-LEAK WARNING: {len(leaked):,} scanned images are labelled "
              f"'notused' in image_type_labels_2ndrev.csv but are still in data/photos.")
        print( "     They were manually rejected in v2. Remove them first, e.g.:")
        print( "     python notebooks/15_apply_screening_removals.py --flagged <list> --apply")
    else:
        print("\n  OK curation-leak guard: no 2nd-rev 'notused' files remain in data/photos.")
else:
    print(f"\n  (curation-leak guard skipped — {LABELS_2NDREV} not found)")

(SPLITS_DIR / "unreadable_images.txt").write_text(
    "\n".join(unreadable), encoding="utf-8")

# ── Apply confirmed taxonomy synonyms (data/taxonomy_synonyms.csv) ─────────────
# Only rows with status=='confirmed' are applied: remap genus AND the species'
# genus-prefix (e.g. 'Trichotosia flexuosa' -> 'Eria flexuosa'). Files are NOT moved;
# only the label changes, so the classifier/retrieval treat them as the accepted genus.
SYN_CSV = PROJECT_ROOT / "data" / "taxonomy_synonyms.csv"
synonyms = {}
if SYN_CSV.exists():
    with open(SYN_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status", "").strip().lower() == "confirmed" and r.get("to_genus", "").strip():
                synonyms[r["from_genus"].strip()] = r["to_genus"].strip()
if synonyms:
    n_remapped = 0
    for rec in all_records:
        g = rec["genus"]
        if g in synonyms:
            tg = synonyms[g]
            rec["genus"] = tg
            parts = rec["species"].split(" ")
            if parts and parts[0] == g:
                parts[0] = tg
                rec["species"] = " ".join(parts)
            n_remapped += 1
    print(f"\n  Applied {len(synonyms)} confirmed synonym(s): "
          f"{', '.join(f'{k}->{v}' for k, v in synonyms.items())} "
          f"({n_remapped:,} images remapped)")
    n_genera  = len(set(r['genus']   for r in all_records))
    n_species = len(set(r['species'] for r in all_records))
    print(f"  After merge: {n_genera} genera, {n_species:,} species")

# ── Stratified split at species level ─────────────────────────────────────────
species_map = defaultdict(list)
for rec in all_records:
    species_map[rec["species"]].append(rec)

genus_species = defaultdict(list)
for sp, recs in species_map.items():
    genus_species[recs[0]["genus"]].append(sp)

train_rows, val_rows, test_rows = [], [], []

def species_rank(sp):
    """Stable per-species hash in [0,1) — deterministic, independent of genus order or
    composition (md5, not Python's salted hash()). Editing one genus therefore cannot
    reshuffle any other genus's split. Coverage rules below still guarantee >=1 val/test."""
    return int(hashlib.md5(sp.encode("utf-8")).hexdigest(), 16) / 16 ** 32

for genus, species_list in sorted(genus_species.items()):
    species_list = sorted(species_list, key=species_rank)   # deterministic, was random.shuffle
    n = len(species_list)
    if n == 1:
        nv, nt = 0, 0                      # cannot split a single species
    elif n == 2:
        nv, nt = 1, 0                      # 1 val, 1 train
    else:                                  # n >= 3 → ensure >=1 val AND >=1 test
        nv = max(1, round(n * VAL_FRAC))
        nt = max(1, round(n * VAL_FRAC))
        nv = min(nv, n - 2)
        nt = min(nt, n - nv - 1)

    for sp in species_list[:nv]:
        val_rows  += [{**r, "split": "val"}  for r in species_map[sp]]
    for sp in species_list[nv:nv + nt]:
        test_rows += [{**r, "split": "test"} for r in species_map[sp]]
    for sp in species_list[nv + nt:]:
        train_rows += [{**r, "split": "train"} for r in species_map[sp]]

tot = len(all_records)
print(f"\n  Train: {len(train_rows):,} ({100*len(train_rows)/tot:.1f}%)")
print(f"  Val:   {len(val_rows):,} ({100*len(val_rows)/tot:.1f}%)")
print(f"  Test:  {len(test_rows):,} ({100*len(test_rows)/tot:.1f}%)")

# ── Genus coverage report (how many genera are evaluable) ──────────────────────
train_g = set(r["genus"] for r in train_rows)
val_g   = set(r["genus"] for r in val_rows)
test_g  = set(r["genus"] for r in test_rows)
train_only = sorted(train_g - val_g - test_g)

cov = [
    "SPLIT COVERAGE REPORT",
    "=" * 40,
    f"Genera in train : {len(train_g)}",
    f"Genera in val   : {len(val_g)}",
    f"Genera in test  : {len(test_g)}",
    f"Train-only genera (single-species, not evaluable): {len(train_only)}",
    "  " + ", ".join(train_only) if train_only else "  (none)",
]
cov_text = "\n".join(cov)
print("\n" + cov_text)
(SPLITS_DIR / "split_coverage_report.txt").write_text(cov_text, encoding="utf-8")

# ── Save CSVs (base + live variants are identical post-curation) ───────────────
FIELDS = ["split", "genus", "species", "path"]

def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in FIELDS})

save_csv(SPLITS_DIR / "train.csv",      train_rows)
save_csv(SPLITS_DIR / "val.csv",        val_rows)
save_csv(SPLITS_DIR / "test.csv",       test_rows)
save_csv(SPLITS_DIR / "all_images.csv", train_rows + val_rows + test_rows)
# Live variants = same set (illustrations/herbarium already curated out)
save_csv(SPLITS_DIR / "train_live.csv", train_rows)
save_csv(SPLITS_DIR / "val_live.csv",   val_rows)
save_csv(SPLITS_DIR / "test_live.csv",  test_rows)

print(f"\nSaved splits to {SPLITS_DIR}")
print("Done.")
