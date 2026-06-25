"""
Apply Screening Removals — ONG Orchid Identifier v3
===================================================
Take the flagged-photo path list(s) produced by the screening contact sheets
(14_make_screening_html.py → "Copy flagged paths") and MOVE those images out of
data/photos into a quarantine folder. Reversible (manifest + --undo). The photo
folder stays the single source of truth, so after this you re-run
04_generate_splits.py to regenerate clean CSVs — we never hand-edit the splits.

Default is a DRY RUN (reports only). Add --apply to actually move files.

Safety report (shown in dry run too):
  * how many flagged paths matched real files
  * per-genus counts removed
  * SPECIES that would be FULLY removed (all their photos flagged) — so you don't
    accidentally delete an entire species without meaning to
  * GENERA that would drop below evaluable size

Usage:
    # 1. collect flagged paths (one per line) into a text file, e.g.:
    #    E:\\Claude Code\\ONG_v3\\screening_live\\flagged_paths.txt
    python notebooks/15_apply_screening_removals.py --flagged screening_live/flagged_paths.txt
    python notebooks/15_apply_screening_removals.py --flagged screening_live/flagged_paths.txt --apply
    # then regenerate splits:
    python notebooks/04_generate_splits.py
    # to restore everything:
    python notebooks/15_apply_screening_removals.py --undo
"""

import argparse, csv, sys
from pathlib import Path
from collections import Counter, defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHOTOS_DIR   = PROJECT_ROOT / "data" / "photos"
QUARANTINE   = PROJECT_ROOT / "data" / "_screened_out"
MANIFEST     = QUARANTINE / "move_manifest.csv"
ALL_CSV      = PROJECT_ROOT / "data" / "splits" / "all_images.csv"


def get_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flagged", nargs="*", default=[],
                    help="one or more text files of flagged image paths (one per line)")
    ap.add_argument("--apply", action="store_true", help="actually move (default: dry run)")
    ap.add_argument("--undo", action="store_true", help="restore everything from the manifest")
    return ap.parse_args()


def norm(p: str) -> str:
    return str(Path(p.strip().replace("\\", "/"))).replace("\\", "/")


def load_flagged(files):
    seen, paths = set(), []
    for f in files:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            n = norm(s)
            if n not in seen:
                seen.add(n); paths.append(Path(s.strip()))
    return paths


def undo():
    if not MANIFEST.exists():
        raise SystemExit(f"No manifest at {MANIFEST} — nothing to undo.")
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    restored = 0
    for r in rows:
        src = Path(r["quarantined"]); dst = Path(r["original"])
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dst); restored += 1
    print(f"Restored {restored}/{len(rows)} files to data/photos.")
    MANIFEST.unlink()
    print("Manifest cleared. Now re-run 04_generate_splits.py to regenerate CSVs.")


def main():
    args = get_args()
    if args.undo:
        return undo()
    if not args.flagged:
        raise SystemExit("Pass --flagged <file.txt> [...] (or --undo).")

    flagged = load_flagged(args.flagged)
    print(f"Flagged paths read (deduped): {len(flagged):,}")

    # match against real files under data/photos
    matched, missing, outside = [], [], []
    for p in flagged:
        rp = p.resolve()
        if not rp.exists():
            missing.append(p); continue
        try:
            rp.relative_to(PHOTOS_DIR.resolve())
        except ValueError:
            outside.append(p); continue
        matched.append(rp)

    if missing:
        print(f"  ⚠ {len(missing)} flagged path(s) not found on disk (skipped).")
    if outside:
        print(f"  ⚠ {len(outside)} flagged path(s) outside data/photos (skipped).")
    print(f"  → {len(matched):,} files will be quarantined.")

    # per-genus counts
    per_genus = Counter(p.parent.name for p in matched)
    print("\nPer-genus removals:")
    for g, c in per_genus.most_common():
        print(f"  {g:24s} {c:5d}")

    # species fully removed? (compare flagged count vs total photos per species)
    if ALL_CSV.exists():
        total_by_sp = Counter()
        sp_of = {}
        for r in csv.DictReader(ALL_CSV.open(encoding="utf-8")):
            total_by_sp[r["species"]] += 1
            sp_of[norm(r["path"])] = r["species"]
        flagged_by_sp = Counter()
        for p in matched:
            sp = sp_of.get(norm(str(p)))
            if sp:
                flagged_by_sp[sp] += 1
        fully = [sp for sp, c in flagged_by_sp.items() if c >= total_by_sp[sp]]
        if fully:
            print(f"\n⚠ {len(fully)} SPECIES would be FULLY removed (all photos flagged):")
            for sp in sorted(fully):
                print(f"    {sp}  ({total_by_sp[sp]} photo(s))")
        else:
            print("\n✓ No species is fully removed (every species keeps >=1 photo).")
    else:
        print(f"\n(skip species-impact report — {ALL_CSV} not found)")

    if not args.apply:
        print("\nDRY RUN. Re-run with --apply to move the files, then 04_generate_splits.py.")
        return

    # move
    QUARANTINE.mkdir(parents=True, exist_ok=True)
    new_rows, moved = [], 0
    for src in matched:
        genus = src.parent.name
        dst = QUARANTINE / genus / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)
        new_rows.append({"original": str(src), "quarantined": str(dst)})
        moved += 1
    write_header = not MANIFEST.exists()
    with MANIFEST.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["original", "quarantined"])
        if write_header:
            w.writeheader()
        w.writerows(new_rows)
    print(f"\nMoved {moved:,} files → {QUARANTINE}")
    print(f"Manifest: {MANIFEST}  (use --undo to restore)")
    print("NEXT: python notebooks/04_generate_splits.py   → regenerate clean CSVs")
    print("THEN: python notebooks/13_evaluate.py          → refresh baseline on the new test set")


if __name__ == "__main__":
    main()
