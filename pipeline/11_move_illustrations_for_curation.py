"""
Move Illustrations / Herbarium → Curation Folder — ONG Orchid Identifier v2

Botanical illustrations (~4,600 line drawings & lithographs) and herbarium
specimens are currently mixed into data/photos/ and pollute classifier training.
This script MOVES them out into a dedicated curation area so they can be
reviewed manually before any retraining.

Source of truth: data/image_type_labels.csv  (the granular label file — has
  'illustration' / 'herbarium' / 'live' / 'uncertain' / 'error'.
  NOTE: image_type_labels_2ndrev.csv only has live/notused and is NOT used here.)

What it does:
  - Reads the label CSV.
  - For every row whose label is in --labels (default: illustration,herbarium),
    MOVES the file from   data/photos/{Genus}/{file}
                    to    data/curation/{label}s/{Genus}/{file}
  - Writes a reversible manifest: data/curation/move_manifest.csv
        (old_path, new_path, genus, label)
  - Builds an HTML contact-sheet per label for fast visual curation:
        data/curation/illustrations_review.html
        data/curation/herbariums_review.html

Safety:
  - DRY-RUN by default: prints per-genus counts and writes nothing.
    Add --apply to actually move files.
  - Reversible: see 11b note below / use the manifest to move everything back.

Usage:
    python notebooks/11_move_illustrations_for_curation.py            # dry-run
    python notebooks/11_move_illustrations_for_curation.py --apply    # move
    python notebooks/11_move_illustrations_for_curation.py --apply --labels illustration
    python notebooks/11_move_illustrations_for_curation.py --undo     # reverse a previous move

After manual curation (you delete true illustrations / move real live photos
back into data/photos/{Genus}/), re-run notebooks/05_detect_image_type.py and
notebooks/04_generate_splits.py to regenerate the live-only splits.
"""

import argparse
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # E:/Claude Code/ONG_v2
DATA_DIR     = PROJECT_ROOT / "data"
PHOTOS_DIR   = DATA_DIR / "photos"
CURATION_DIR = DATA_DIR / "curation"
LABELS_CSV   = DATA_DIR / "image_type_labels.csv"
MANIFEST_CSV = CURATION_DIR / "move_manifest.csv"
# ─────────────────────────────────────────────────────────────────────────────


def resolve_src(raw_path: str) -> Path:
    """CSV stores Windows-relative paths like 'data\\photos\\Genus\\file.jpg'.
    Normalise separators and resolve relative to the project root."""
    norm = raw_path.replace("\\", "/").lstrip("/")
    p = Path(norm)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def dest_for(label: str, genus: str, filename: str) -> Path:
    """Curation destination: data/curation/{label}s/{Genus}/{file}."""
    return CURATION_DIR / f"{label}s" / genus / filename


def build_html(label: str, entries: list[dict]) -> None:
    """Write a thumbnail contact-sheet so curation can be done by eye.
    Image src is relative to the HTML file (which lives in data/curation/)."""
    out = CURATION_DIR / f"{label}s_review.html"
    by_genus: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_genus[e["genus"]].append(e)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Curation: {label}s ({len(entries)})</title>",
        "<style>",
        "body{font-family:Lato,Arial,sans-serif;background:#1d2b22;color:#eee;margin:0;padding:16px}",
        "h1{font-size:20px} h2{font-size:15px;color:#9ad27f;border-bottom:1px solid #3a5043;"
        "padding-bottom:4px;margin-top:28px}",
        ".grid{display:flex;flex-wrap:wrap;gap:8px}",
        ".card{width:170px;font-size:10px;background:#26352b;border-radius:6px;padding:4px;"
        "word-break:break-all}",
        ".card img{width:160px;height:160px;object-fit:contain;background:#000;border-radius:4px}",
        "</style></head><body>",
        f"<h1>Manual curation — <b>{label}s</b>: {len(entries)} files</h1>",
        "<p>Hapus yang benar-benar bukan foto hidup. Untuk yang ternyata foto hidup, "
        "pindahkan kembali file-nya ke <code>data/photos/{Genus}/</code> "
        "(atau gunakan <code>--undo</code> dengan manifest).</p>",
    ]
    for genus in sorted(by_genus):
        items = by_genus[genus]
        parts.append(f"<h2>{genus} ({len(items)})</h2><div class='grid'>")
        for e in items:
            # new_path is absolute; make it relative to CURATION_DIR for the <img src>
            try:
                rel = Path(e["new_path"]).resolve().relative_to(CURATION_DIR.resolve())
                src = str(rel).replace("\\", "/")
            except ValueError:
                src = e["new_path"]
            parts.append(
                f"<div class='card'><img loading='lazy' src='{src}'><br>{e['filename']}</div>"
            )
        parts.append("</div>")
    parts.append("</body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"  HTML review sheet: {out}")


def do_undo() -> None:
    """Reverse a previous move using the manifest."""
    if not MANIFEST_CSV.exists():
        print(f"ERROR: no manifest at {MANIFEST_CSV} — nothing to undo.")
        return
    rows = list(csv.DictReader(MANIFEST_CSV.open(encoding="utf-8")))
    moved = missing = 0
    for r in rows:
        new_p, old_p = Path(r["new_path"]), Path(r["old_path"])
        if new_p.exists():
            old_p.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(new_p), str(old_p))
            moved += 1
        else:
            missing += 1
    print(f"Undo complete: {moved} restored, {missing} not found "
          f"(already curated/deleted).")
    print(f"Manifest left in place: {MANIFEST_CSV}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="Actually move files (default is dry-run).")
    ap.add_argument("--undo", action="store_true",
                    help="Reverse a previous move using move_manifest.csv.")
    ap.add_argument("--labels", default="illustration,herbarium",
                    help="Comma-separated labels to move (default: illustration,herbarium).")
    ap.add_argument("--csv", default=str(LABELS_CSV),
                    help="Label CSV to read (default: data/image_type_labels.csv).")
    args = ap.parse_args()

    if args.undo:
        do_undo()
        return

    labels_to_move = {s.strip() for s in args.labels.split(",") if s.strip()}
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: label CSV not found: {csv_path}")
        return

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    targets = [r for r in rows if r["label"] in labels_to_move]

    # ── Per-genus / per-label summary ────────────────────────────────────────
    per_label = Counter(r["label"] for r in targets)
    per_genus = defaultdict(Counter)
    for r in targets:
        per_genus[r["genus"]][r["label"]] += 1

    print("=" * 64)
    print(f"{'DRY-RUN — no files moved' if not args.apply else 'APPLYING MOVE'}")
    print("=" * 64)
    print(f"Source CSV : {csv_path}")
    print(f"Labels     : {', '.join(sorted(labels_to_move))}")
    print(f"Total files: {len(targets):,}")
    for lbl in sorted(per_label):
        print(f"  {lbl:<14} {per_label[lbl]:>6,}")
    print("\nTop 20 genera by files to move:")
    print(f"  {'Genus':<32} " + " ".join(f"{l:>12}" for l in sorted(labels_to_move)) + f" {'total':>7}")
    print("  " + "-" * 60)
    for genus in sorted(per_genus, key=lambda g: sum(per_genus[g].values()), reverse=True)[:20]:
        c = per_genus[genus]
        cols = " ".join(f"{c.get(l, 0):>12,}" for l in sorted(labels_to_move))
        print(f"  {genus:<32} {cols} {sum(c.values()):>7,}")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to move these files.")
        return

    # ── Move + build manifest ────────────────────────────────────────────────
    CURATION_DIR.mkdir(parents=True, exist_ok=True)
    manifest, moved, missing, html_entries = [], 0, 0, defaultdict(list)
    for r in targets:
        src = resolve_src(r["path"])
        dst = dest_for(r["label"], r["genus"], r["filename"])
        if not src.exists():
            missing += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Avoid clobbering an existing file at the destination
        if dst.exists():
            stem, suf = dst.stem, dst.suffix
            n = 1
            while dst.exists():
                dst = dst.with_name(f"{stem}__dup{n}{suf}")
                n += 1
        shutil.move(str(src), str(dst))
        rec = {"old_path": str(src), "new_path": str(dst),
               "genus": r["genus"], "label": r["label"]}
        manifest.append(rec)
        html_entries[r["label"]].append({"genus": r["genus"],
                                         "filename": dst.name,
                                         "new_path": str(dst)})
        moved += 1

    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["old_path", "new_path", "genus", "label"])
        w.writeheader()
        w.writerows(manifest)

    print(f"\nMoved   : {moved:,}")
    print(f"Missing : {missing:,} (path in CSV not found on disk — skipped)")
    print(f"Manifest: {MANIFEST_CSV}  (use --undo to reverse)")

    for lbl in sorted(html_entries):
        build_html(lbl, html_entries[lbl])

    print("\nNext: curate visually, then re-run "
          "05_detect_image_type.py and 04_generate_splits.py.")


if __name__ == "__main__":
    main()
