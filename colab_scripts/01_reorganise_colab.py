"""
Step 1: Verify Dataset Structure — Colab Version
Run via: %run /content/drive/MyDrive/orchid_project/scripts/01_reorganise_colab.py

The merged dataset (photos.zip) is already organised per genus.
Sources: BH Lucid, ONG website, Jeffrey Champion, Kurt Metzger,
         iNaturalist RG (_RG), iNaturalist Needs-ID (_NID)

This script just verifies the structure and prints a summary.
No reorganisation needed — photos are ready for training directly.
"""

import os
from pathlib import Path
from collections import defaultdict

PHOTOS_DIR = Path("/content/photos")

print("Verifying merged dataset structure...")
genus_counts = defaultdict(int)
total = 0

for genus_dir in sorted(PHOTOS_DIR.iterdir()):
    if not genus_dir.is_dir():
        continue
    imgs = [f for f in genus_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg",".jpeg",".png"}]
    genus_counts[genus_dir.name] = len(imgs)
    total += len(imgs)

print(f"\nTotal genera : {len(genus_counts)}")
print(f"Total photos : {total:,}  (expected 28,718)")
print(f"\nTop 10 genera:")
for g, n in sorted(genus_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  {g:<30} {n:>5}")

if total == 28718:
    print("\nDataset OK — ready for training.")
else:
    print(f"\nWARNING: expected 28,718 photos but found {total:,}")
    print("  Sources: BH Lucid (6,086) + ONG (12,434) + Jeffrey Champion (4,269)")
    print("           + Kurt Metzger (3,345) + iNat RG (1,283) + iNat NID (1,301)")
