"""
Screening contact-sheets — ONG Orchid Identifier v3
===================================================
Build browsable HTML contact sheets of EVERY live image that will go into training
(train_live + val_live + test_live), so you can eyeball the dataset before a multi-hour
Colab run. One page per genus + an index. Images are referenced by relative path
(not copied/embedded), so generation is instant and the folder stays tiny.

Screening workflow in the browser:
  * Click any thumbnail to FLAG it (red border). Click again to un-flag.
  * "Copy flagged paths" button → clipboard has the newline-separated absolute paths
    of everything you flagged → paste into a text file to act on later.
  * Caption under each photo = species · split · filename.

Run:
    python notebooks/14_make_screening_html.py
    # then open screening_live/index.html in a browser

Options:
    --min-count N   only genera with >= N images (skip tiny ones)
    --thumb PX      thumbnail width in px (default 170)
    --out DIR       output folder (default: screening_live)
"""

import argparse, html, sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")   # allow unicode prints on Windows cp1252
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS = PROJECT_ROOT / "data" / "splits"


def get_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-count", type=int, default=0)
    ap.add_argument("--thumb", type=int, default=170)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "screening_live"))
    return ap.parse_args()


def load_live():
    frames = []
    for s in ("train_live", "val_live", "test_live"):
        d = pd.read_csv(SPLITS / f"{s}.csv")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["fname"] = df["path"].apply(lambda p: p.replace("\\", "/").split("/")[-1])
    return df


def rel_src(out_dir: Path, abspath: str) -> str:
    """Relative POSIX URL from a per-genus html (in out_dir) to the image on disk."""
    p = Path(abspath.replace("\\", "/"))
    try:
        rel = Path(p).resolve().relative_to(out_dir.parent.resolve())
        rel = ".." + "/" + rel.as_posix()          # out_dir is one level under project root
    except ValueError:
        rel = p.as_posix()                          # fallback: absolute
    return quote(rel, safe="/:")


PAGE_CSS = """
<style>
:root{--bg:#f4f6f4;--card:#fff;--ink:#1f2a1f;--accent:#2e6b2e;--flag:#c0392b}
*{box-sizing:border-box}
body{margin:0;font-family:Lato,Arial,sans-serif;background:var(--bg);color:var(--ink)}
header{position:sticky;top:0;background:var(--accent);color:#fff;padding:10px 16px;
  display:flex;gap:16px;align-items:center;flex-wrap:wrap;z-index:5}
header h1{font-size:18px;margin:0}
header .meta{font-size:13px;opacity:.9}
button{background:#fff;color:var(--accent);border:0;border-radius:6px;padding:7px 12px;
  font-weight:700;cursor:pointer}
a.back{color:#fff;text-decoration:underline}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(THUMBpx,1fr));
  gap:10px;padding:14px}
figure{margin:0;background:var(--card);border:3px solid transparent;border-radius:8px;
  overflow:hidden;cursor:pointer;box-shadow:0 1px 3px rgba(0,0,0,.12)}
figure.flagged{border-color:var(--flag);box-shadow:0 0 0 2px var(--flag)}
figure img{width:100%;height:THUMBpx;object-fit:cover;display:block;background:#ddd}
figcaption{font-size:11px;line-height:1.25;padding:5px 6px;word-break:break-word}
figcaption .sp{font-style:italic;color:var(--accent)}
figcaption .sl{color:#888}
#count{font-weight:700}
</style>
"""

PAGE_JS = """
<script>
function toggle(el){el.classList.toggle('flagged');upd();}
function upd(){document.getElementById('count').textContent=
  document.querySelectorAll('figure.flagged').length;}
function copyFlagged(){
  const ps=[...document.querySelectorAll('figure.flagged')].map(f=>f.dataset.path);
  if(!ps.length){alert('No photos flagged yet.');return;}
  navigator.clipboard.writeText(ps.join('\\n')).then(()=>
    alert(ps.length+' path(s) copied to clipboard.'));
}
</script>
"""


def genus_page(out_dir: Path, genus: str, rows: pd.DataFrame, thumb: int):
    cards = []
    for r in rows.itertuples(index=False):
        src = rel_src(out_dir, r.path)
        sp = html.escape(str(r.species)); fn = html.escape(str(r.fname))
        sl = html.escape(str(r.split)); ap = html.escape(str(r.path))
        cards.append(
            f'<figure data-path="{ap}" onclick="toggle(this)">'
            f'<img loading="lazy" src="{src}" alt="{fn}">'
            f'<figcaption><span class="sp">{sp}</span><br>'
            f'<span class="sl">{sl}</span> · {fn}</figcaption></figure>'
        )
    head = (f'<header><h1>{html.escape(genus)}</h1>'
            f'<span class="meta">{len(rows):,} live photos · '
            f'{rows.species.nunique()} species</span>'
            f'<a class="back" href="index.html">&larr; index</a>'
            f'<button onclick="copyFlagged()">Copy flagged paths</button>'
            f'<span class="meta">flagged: <span id="count">0</span></span></header>')
    css = PAGE_CSS.replace("THUMB", str(thumb))
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<title>{html.escape(genus)} — screening</title>{css}</head><body>'
           f'{head}<div class="grid">{"".join(cards)}</div>{PAGE_JS}</body></html>')
    (out_dir / f"{genus}.html").write_text(doc, encoding="utf-8")


def index_page(out_dir: Path, df: pd.DataFrame, thumb: int):
    counts = df.groupby("genus").agg(n=("path", "size"),
                                     sp=("species", "nunique")).sort_values("n", ascending=False)
    first = df.groupby("genus").first()["path"]
    cards = []
    for genus, row in counts.iterrows():
        src = rel_src(out_dir, first[genus])
        cards.append(
            f'<a class="gcard" href="{quote(genus)}.html">'
            f'<img loading="lazy" src="{src}" alt="{html.escape(genus)}">'
            f'<div class="gname">{html.escape(genus)}</div>'
            f'<div class="gmeta">{int(row.n):,} photos · {int(row.sp)} sp</div></a>'
        )
    css = PAGE_CSS.replace("THUMB", str(thumb)) + """
<style>
.grid{grid-template-columns:repeat(auto-fill,minmax(150px,1fr))}
a.gcard{display:block;background:#fff;border-radius:8px;overflow:hidden;text-decoration:none;
  color:var(--ink);box-shadow:0 1px 3px rgba(0,0,0,.12)}
a.gcard img{width:100%;height:120px;object-fit:cover;display:block;background:#ddd}
.gname{font-weight:700;padding:6px 8px 0}
.gmeta{font-size:12px;color:#888;padding:0 8px 8px}
</style>"""
    head = (f'<header><h1>Live dataset screening</h1>'
            f'<span class="meta">{len(df):,} photos · {df.genus.nunique()} genera · '
            f'{df.species.nunique():,} species — click a genus</span></header>')
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<title>Live dataset screening — index</title>{css}</head><body>'
           f'{head}<div class="grid">{"".join(cards)}</div></body></html>')
    (out_dir / "index.html").write_text(doc, encoding="utf-8")


def main():
    args = get_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_live()
    if args.min_count:
        keep = df.groupby("genus")["path"].transform("size") >= args.min_count
        df = df[keep].reset_index(drop=True)
    print(f"Live images: {len(df):,} | genera: {df.genus.nunique()} | "
          f"species: {df.species.nunique():,}")
    for genus, rows in df.groupby("genus"):
        genus_page(out_dir, genus, rows.reset_index(drop=True), args.thumb)
    index_page(out_dir, df, args.thumb)
    print(f"Done → open {out_dir / 'index.html'} in a browser.")
    print("Click thumbnails to flag, then 'Copy flagged paths' to export the list.")


if __name__ == "__main__":
    main()
