"""
prepare_hf_space_v3.py — Run LOCALLY before pushing the v3 Space.

Maps each FAISS metadata record to a bundled thumbnail (reusing the v2 thumbnails,
since v3 reuses the same photo set), then writes metadata back in place.

Prerequisites — place these inside this folder first:
  models/dinov2l/best_model.pth      (= best_model_global.pth downloaded from Drive)
  models/dinov2l/vocab.json
  models/ong_species_index.faiss     (= retrieval_global/species_index.faiss from Drive)
  models/ong_metadata.json           (= retrieval_global/metadata.json from Drive)
  models/genus_wpa_counts.json       (copy from the v2 Space)
  thumbnails/                        (copy the whole folder from the v2 Space, OR let
                                      this script copy it via --v2-thumbs)

Run:
  cd "E:/Claude Code/ONG_v3/hf_space/ong-orchid-identifier-v3"
  python prepare_hf_space_v3.py
  # optionally copy v2 thumbnails first:
  python prepare_hf_space_v3.py --v2-thumbs "E:/Claude Code/ONG_v2/hf_space/ong-orchid-identifierv2/thumbnails"
"""

import argparse
import json
import shutil
from pathlib import Path

SPACE_DIR  = Path(__file__).parent
MODELS_DIR = SPACE_DIR / "models"
THUMB_DIR  = SPACE_DIR / "thumbnails"
META_PATH  = MODELS_DIR / "ong_metadata.json"


def thumb_rel_for(rec: dict) -> str:
    """thumbnails/<Genus>/<filename.jpg> from the record's source path."""
    src = str(rec.get("path", "")).replace("\\", "/")
    if not src:
        return ""
    filename = src.rsplit("/", 1)[-1]
    genus = rec.get("genus", "")
    return f"thumbnails/{genus}/{filename}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-thumbs", default="", help="path to v2 thumbnails/ to copy in if missing")
    args = ap.parse_args()

    required = [
        MODELS_DIR / "dinov2l" / "best_model.pth",
        MODELS_DIR / "dinov2l" / "vocab.json",
        MODELS_DIR / "ong_species_index.faiss",
        META_PATH,
        MODELS_DIR / "genus_wpa_counts.json",
    ]
    missing = [str(f) for f in required if not f.exists()]
    if missing:
        print("[ERROR] Missing files — download from Drive / copy from v2 first:")
        for m in missing:
            print("  ", m)
        return

    if args.v2_thumbs and not THUMB_DIR.exists():
        print(f"Copying thumbnails from {args.v2_thumbs} ...")
        shutil.copytree(args.v2_thumbs, THUMB_DIR)
    if not THUMB_DIR.exists():
        print("[ERROR] thumbnails/ not found — copy it from the v2 Space (or use --v2-thumbs).")
        return

    with open(META_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Metadata records: {len(metadata):,}")

    ok = miss = 0
    for rec in metadata:
        rel = thumb_rel_for(rec)
        if rel and (SPACE_DIR / rel).exists():
            rec["thumb_path"] = rel
            ok += 1
        else:
            rec["thumb_path"] = ""
            miss += 1

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, separators=(",", ":"))

    print(f"thumb_path resolved : {ok:,}")
    print(f"thumb_path missing  : {miss:,}  (will show a placeholder)")
    print(f"\nMetadata saved → {META_PATH}")
    print("\nReady. Next:")
    print("  1. Create a new Gradio Space on huggingface.co")
    print("  2. git clone it, copy ALL files from this folder in")
    print("  3. Set the HF_TOKEN secret in Space settings (for community uploads)")
    print("  4. git add . && git commit && git push")
    print("  5. Update the iframe src on birdsheadorchid.id to the new Space URL")


if __name__ == "__main__":
    main()
