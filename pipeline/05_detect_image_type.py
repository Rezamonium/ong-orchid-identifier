"""
Image Type Detector — ONG Orchid Identifier v2
Classifies photos in data/photos/ into:
  - live       : live orchid photo (field/cultivation) — KEEP
  - herbarium  : dried specimen on white background    — EXCLUDE from FAISS
  - illustration: botanical line drawing               — EXCLUDE from training
  - uncertain  : borderline cases for manual review

Detection uses per-pixel colour statistics (no ML required):
  white_pct     — fraction of pixels with R,G,B all > 230
  light_pct     — fraction of pixels with mean brightness > 210
  high_sat_pct  — fraction of pixels with HSV saturation > 0.25 (vivid colours)
  mean_sat      — mean HSV saturation across all pixels

Rules (conservative — errs toward "uncertain" rather than false exclusion):
  illustration : white_pct > 0.55  AND high_sat_pct < 0.04
  herbarium    : light_pct > 0.38  AND high_sat_pct < 0.09  (and not illustration)
  live         : high_sat_pct >= 0.09
  uncertain    : everything else

Output:
  data/image_type_labels.csv    — every photo labelled
  data/image_type_report.txt    — summary + thresholds
  data/splits/train_live.csv    — train split, live only
  data/splits/val_live.csv
  data/splits/test_live.csv
"""

import csv, random
from pathlib import Path
from collections import defaultdict, Counter

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("ERROR: Install Pillow and numpy first: pip install Pillow numpy")
    raise

# ── Config ────────────────────────────────────────────────────────────────────
PHOTOS_DIR  = Path("E:/Claude Code/ONG_v2/data/photos")
DATA_DIR    = Path("E:/Claude Code/ONG_v2/data")
SPLITS_DIR  = DATA_DIR / "splits"
VALID_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Thresholds — tuned after visual validation of uncertain cases
ILLUST_WHITE_PCT    = 0.55   # >55% very white pixels (pure B&W line drawings)
ILLUST_HIGH_SAT_PCT = 0.04   # <4% vivid-colour pixels
ILLUST_MUTED_LIGHT  = 0.25   # coloured vintage illustrations: light bg but muted colours
ILLUST_MUTED_SAT    = 0.05   # max high_sat for vintage/coloured illustrations
HERB_LIGHT_PCT      = 0.30   # >30% light pixels (lowered to catch beige/tan herbarium sheets)
HERB_HIGH_SAT_PCT   = 0.09   # <9% vivid-colour pixels (coloured labels on herbarium are < 9%)
LIVE_HIGH_SAT_PCT   = 0.07   # >=7% vivid-colour pixels = live (lowered from 0.09)

SAMPLE_SIZE = 0              # 0 = skip validation, run full scan directly
RESIZE_TO   = (128, 128)     # downsample for speed (keep proportions)
# ─────────────────────────────────────────────────────────────────────────────

def get_image_stats(img_path: Path) -> dict:
    """Return colour statistics for one image."""
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(RESIZE_TO, Image.LANCZOS)
        rgb = np.array(img, dtype=np.float32) / 255.0   # shape (H, W, 3)

        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

        # White pixels: all channels very bright
        white_mask  = (r > 0.90) & (g > 0.90) & (b > 0.90)
        white_pct   = white_mask.mean()

        # Light pixels: mean brightness high
        brightness  = (r + g + b) / 3.0
        light_pct   = (brightness > 0.82).mean()

        # HSV saturation approximation (fast, no colorsys loop)
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin
        # saturation = delta/cmax where cmax>0, else 0
        sat = np.where(cmax > 0.01, delta / (cmax + 1e-8), 0.0)

        mean_sat      = sat.mean()
        high_sat_pct  = (sat > 0.25).mean()   # vivid colours

        return {
            "white_pct":    round(float(white_pct),   4),
            "light_pct":    round(float(light_pct),   4),
            "mean_sat":     round(float(mean_sat),    4),
            "high_sat_pct": round(float(high_sat_pct),4),
            "ok": True,
        }
    except Exception as e:
        return {"white_pct":0,"light_pct":0,"mean_sat":0,"high_sat_pct":0,"ok":False,"error":str(e)}


def classify(stats: dict) -> str:
    if not stats["ok"]:
        return "error"
    w  = stats["white_pct"]
    lp = stats["light_pct"]
    hs = stats["high_sat_pct"]
    ms = stats["mean_sat"]

    # Pure B&W line drawings (white bg + black lines, no colour)
    if w > ILLUST_WHITE_PCT and hs < ILLUST_HIGH_SAT_PCT:
        return "illustration"
    # Vintage coloured illustrations (lithographs): light bg, very muted colour
    if lp > ILLUST_MUTED_LIGHT and hs < ILLUST_MUTED_SAT:
        return "illustration"
    # Herbarium specimens: light background, low vivid colour
    # Extra check: exclude herbarium with coloured labels (ms slightly higher)
    if lp > HERB_LIGHT_PCT and hs < HERB_HIGH_SAT_PCT:
        return "herbarium"
    # Live photos: enough vivid colour present
    if hs >= LIVE_HIGH_SAT_PCT:
        return "live"
    # Remaining uncertain: mostly B&W live photos — treat as live
    # (low colour due to B&W photography, not herbarium/illustration)
    return "uncertain"


# ── Step 1: Collect all images ────────────────────────────────────────────────
print("Collecting images...")
all_paths = []
for genus_dir in sorted(PHOTOS_DIR.iterdir()):
    if not genus_dir.is_dir(): continue
    for f in sorted(genus_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in VALID_EXTS:
            all_paths.append(f)

print(f"  Total: {len(all_paths):,} images")

# ── Step 2 (optional): Validate on random sample first ───────────────────────
if SAMPLE_SIZE > 0:
    print(f"\nValidation sample ({SAMPLE_SIZE} random photos):")
    print(f"  {'File':<55} {'white':>6} {'light':>6} {'hi_sat':>7} {'label'}")
    print("  " + "-" * 85)
    sample = random.sample(all_paths, min(SAMPLE_SIZE, len(all_paths)))
    sample_results = []
    for p in sorted(sample):
        stats = get_image_stats(p)
        label = classify(stats)
        short = f"{p.parent.name}/{p.name}"[-54:]
        print(f"  {short:<55} {stats['white_pct']:>6.3f} {stats['light_pct']:>6.3f} "
              f"{stats['high_sat_pct']:>7.3f} {label}")
        sample_results.append((p, stats, label))

    print("\nSample summary:")
    sample_counts = Counter(label for _, _, label in sample_results)
    for lbl, cnt in sorted(sample_counts.items()):
        print(f"  {lbl:<15} {cnt:>4}  ({100*cnt/len(sample_results):.0f}%)")

    ans = input("\nDo the labels look correct? Press Enter to continue with full scan, or Ctrl+C to abort: ")

# ── Step 3: Classify all images ───────────────────────────────────────────────
print(f"\nClassifying all {len(all_paths):,} images...")
rows = []
for i, img_path in enumerate(all_paths, 1):
    stats  = get_image_stats(img_path)
    label  = classify(stats)
    genus  = img_path.parent.name
    rows.append({
        "path":         str(img_path),
        "genus":        genus,
        "filename":     img_path.name,
        "label":        label,
        "white_pct":    stats["white_pct"],
        "light_pct":    stats["light_pct"],
        "mean_sat":     stats["mean_sat"],
        "high_sat_pct": stats["high_sat_pct"],
    })
    if i % 1000 == 0:
        print(f"  [{i:,}/{len(all_paths):,}]...")

# ── Step 4: Save labels CSV ───────────────────────────────────────────────────
labels_csv = DATA_DIR / "image_type_labels.csv"
fields = ["path","genus","filename","label","white_pct","light_pct","mean_sat","high_sat_pct"]
with open(labels_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows)
print(f"\nSaved: {labels_csv}")

# ── Step 5: Summary ───────────────────────────────────────────────────────────
total    = len(rows)
by_label = Counter(r["label"] for r in rows)
by_genus = defaultdict(lambda: Counter())
for r in rows:
    by_genus[r["genus"]][r["label"]] += 1

report_lines = [
    "=" * 60,
    "IMAGE TYPE DETECTION REPORT",
    "=" * 60,
    f"Total photos scanned : {total:,}",
    "",
    "--- OVERALL ---",
]
for lbl in ["live","herbarium","illustration","uncertain","error"]:
    n = by_label.get(lbl, 0)
    report_lines.append(f"  {lbl:<15} {n:>6,}  ({100*n/total:.1f}%)")

report_lines += [
    "",
    "--- THRESHOLDS USED ---",
    f"  illustration (B&W)    : white_pct > {ILLUST_WHITE_PCT} AND high_sat_pct < {ILLUST_HIGH_SAT_PCT}",
    f"  illustration (colour) : light_pct > {ILLUST_MUTED_LIGHT} AND high_sat_pct < {ILLUST_MUTED_SAT}",
    f"  herbarium             : light_pct > {HERB_LIGHT_PCT} AND high_sat_pct < {HERB_HIGH_SAT_PCT} (beige/tan sheets included)",
    f"  live                  : high_sat_pct >= {LIVE_HIGH_SAT_PCT}",
    f"  uncertain             : B&W live photos and edge cases",
    "",
    "--- PER GENUS (genera with most non-live photos) ---",
    f"  {'Genus':<35} {'live':>5} {'herb':>5} {'illus':>6} {'uncert':>7} {'total':>6}",
    "  " + "-" * 65,
]

# Sort genera by non-live count descending
def non_live(g): return sum(v for k,v in by_genus[g].items() if k != "live")
for genus in sorted(by_genus.keys(), key=non_live, reverse=True)[:30]:
    c = by_genus[genus]
    report_lines.append(
        f"  {genus:<35} {c.get('live',0):>5} {c.get('herbarium',0):>5} "
        f"{c.get('illustration',0):>6} {c.get('uncertain',0):>7} "
        f"{sum(c.values()):>6}"
    )

report_text = "\n".join(report_lines)
print("\n" + report_text)

with open(DATA_DIR / "image_type_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

# ── Step 6: Generate filtered splits (live only) ──────────────────────────────
print("\nGenerating live-only split CSVs...")

live_set = {r["path"] for r in rows if r["label"] == "live"}

for split_name in ["train", "val", "test"]:
    src_csv = SPLITS_DIR / f"{split_name}.csv"
    dst_csv = SPLITS_DIR / f"{split_name}_live.csv"
    if not src_csv.exists():
        print(f"  WARNING: {src_csv} not found — skipping")
        continue
    kept, total_split = 0, 0
    with open(src_csv, encoding="utf-8") as fin, \
         open(dst_csv, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            total_split += 1
            if row["path"] in live_set:
                writer.writerow(row)
                kept += 1
    print(f"  {split_name}: {kept:,} / {total_split:,} kept ({100*kept/total_split:.1f}%)")

print(f"\nSaved: {DATA_DIR / 'image_type_report.txt'}")
print("Done.")
