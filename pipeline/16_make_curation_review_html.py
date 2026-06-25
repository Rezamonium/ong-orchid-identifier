"""
Curation Review contact-sheets — ONG Orchid Identifier v3
=========================================================
Visualise the FINAL curation decision on every current live image by combining:
  1. image_type_labels_2ndrev.csv  — the user's manual 2nd-revision labels
     (label == 'notused' → remove).
  2. screening_live/flagged_paths.txt — photos flagged in the new screening pass.

Every photo is classified KEEP or REMOVE; the page pre-marks REMOVE (red) and shows
its source (2ndrev / flagged / both). Three view filters (All / Keep only / Remove only)
let you eyeball the resulting clean set or spot-check the removals. Click a photo to
override its decision; "Copy REMOVE paths" exports the current removal list for this genus.

Nothing is moved/deleted here — this is review only. To apply, feed the union to
15_apply_screening_removals.py, then 04_generate_splits.py.

Run:
    python notebooks/16_make_curation_review_html.py
    # open screening_curation/index.html
"""

import argparse, html, sys
from pathlib import Path
from urllib.parse import quote
from collections import defaultdict

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS   = PROJECT_ROOT / "data" / "splits"
DEF_LABELS  = Path(r"E:\Claude Code\ONG_v2\data\image_type_labels_2ndrev.csv")
DEF_FLAGGED = PROJECT_ROOT / "screening_live" / "flagged_paths.txt"


def get_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(DEF_LABELS))
    ap.add_argument("--flagged", default=str(DEF_FLAGGED))
    ap.add_argument("--thumb", type=int, default=170)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "screening_curation"))
    return ap.parse_args()


def load(args):
    cur = pd.read_csv(SPLITS / "all_images.csv")
    cur["fname"] = cur["path"].str.replace("\\", "/", regex=False).str.split("/").str[-1]
    # 2ndrev manual labels, keyed on genus+filename
    lab = pd.read_csv(args.labels)
    lab["key"] = lab["genus"] + "||" + lab["filename"]
    label_map = dict(zip(lab["key"], lab["label"]))
    cur["label2nd"] = (cur["genus"] + "||" + cur["fname"]).map(label_map).fillna("live")
    # flagged filenames
    flagged = set()
    fp = Path(args.flagged)
    if fp.exists():
        for line in fp.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s:
                flagged.add(Path(s.replace("\\", "/")).name)
    cur["flagged"] = cur["fname"].isin(flagged)
    # status
    def status(r):
        nu = r["label2nd"] == "notused"
        if nu and r["flagged"]:
            return "both"
        if r["flagged"]:
            return "flag"
        if nu:
            return "notused"
        return "keep"
    cur["status"] = cur.apply(status, axis=1)
    return cur, len(flagged)


def rel_src(out_dir: Path, abspath: str) -> str:
    p = Path(abspath.replace("\\", "/"))
    try:
        rel = ".." + "/" + Path(p).resolve().relative_to(out_dir.parent.resolve()).as_posix()
    except ValueError:
        rel = p.as_posix()
    return quote(rel, safe="/:")


BADGE = {"both": "2ndrev+flag", "flag": "flagged", "notused": "2ndrev", "keep": "keep"}

CSS = """
<style>
:root{--bg:#f4f6f4;--card:#fff;--ink:#1f2a1f;--accent:#2e6b2e;--rm:#c0392b;--nu:#d98014}
*{box-sizing:border-box}
body{margin:0;font-family:Lato,Arial,sans-serif;background:var(--bg);color:var(--ink)}
header{position:sticky;top:0;background:var(--accent);color:#fff;padding:10px 16px;
  display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:5}
header h1{font-size:18px;margin:0}
.meta{font-size:13px;opacity:.92}
button{background:#fff;color:var(--accent);border:0;border-radius:6px;padding:7px 11px;
  font-weight:700;cursor:pointer}
button.on{background:#cdebcd}
a.back{color:#fff;text-decoration:underline}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(THUMBpx,1fr));gap:10px;padding:14px}
figure{margin:0;background:var(--card);border:3px solid transparent;border-radius:8px;
  overflow:hidden;cursor:pointer;box-shadow:0 1px 3px rgba(0,0,0,.12)}
figure img{width:100%;height:THUMBpx;object-fit:cover;display:block;background:#ddd}
figcaption{font-size:11px;line-height:1.25;padding:4px 6px;word-break:break-word}
figcaption .sp{font-style:italic;color:var(--accent)}
figcaption .bd{display:inline-block;font-weight:700;font-size:10px;padding:1px 5px;
  border-radius:8px;color:#fff;margin-top:2px}
/* removal states */
figure.rm{opacity:.55}
figure.rm img{filter:grayscale(.3)}
figure[data-status="notused"].rm{border-color:var(--nu)}
figure[data-status="flag"].rm,figure[data-status="both"].rm{border-color:var(--rm)}
.bd.notused{background:var(--nu)} .bd.flag,.bd.both{background:var(--rm)} .bd.keep{background:#7aa37a}
/* view filters */
body.only-keep figure.rm{display:none}
body.only-rm figure:not(.rm){display:none}
</style>
"""

JS = """
<script>
function setView(v,btn){
  document.body.className=v;
  document.querySelectorAll('header .vbtn').forEach(b=>b.classList.remove('on'));
  if(btn)btn.classList.add('on');
}
function toggle(el){el.classList.toggle('rm');upd();}
function upd(){
  document.getElementById('nrm').textContent=document.querySelectorAll('figure.rm').length;
  document.getElementById('nkp').textContent=document.querySelectorAll('figure:not(.rm)').length;
}
function copyRemove(){
  const ps=[...document.querySelectorAll('figure.rm')].map(f=>f.dataset.path);
  if(!ps.length){alert('Nothing marked for removal.');return;}
  navigator.clipboard.writeText(ps.join('\\n')).then(()=>alert(ps.length+' REMOVE path(s) copied.'));
}
</script>
"""


def genus_page(out_dir, genus, rows, thumb):
    fig = []
    for r in rows.itertuples(index=False):
        src = rel_src(out_dir, r.path)
        rm = r.status != "keep"
        cls = "rm" if rm else ""
        bd = BADGE[r.status]
        fig.append(
            f'<figure class="{cls}" data-status="{r.status}" data-path="{html.escape(str(r.path))}" '
            f'onclick="toggle(this)"><img loading="lazy" src="{src}" alt="">'
            f'<figcaption><span class="sp">{html.escape(str(r.species))}</span><br>'
            f'<span class="bd {r.status}">{bd}</span></figcaption></figure>'
        )
    n = len(rows); nrm = int((rows.status != "keep").sum()); nkp = n - nrm
    css = CSS.replace("THUMB", str(thumb))
    head = (
        f'<header><h1>{html.escape(genus)}</h1>'
        f'<span class="meta">{n:,} photos · keep <b id="nkp">{nkp}</b> · '
        f'remove <b id="nrm">{nrm}</b></span>'
        f'<a class="back" href="index.html">&larr; index</a>'
        f'<button class="vbtn on" onclick="setView(\'\',this)">All</button>'
        f'<button class="vbtn" onclick="setView(\'only-keep\',this)">Keep only</button>'
        f'<button class="vbtn" onclick="setView(\'only-rm\',this)">Remove only</button>'
        f'<button onclick="copyRemove()">Copy REMOVE paths</button></header>'
    )
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<title>{html.escape(genus)} — curation review</title>{css}</head><body>'
           f'{head}<div class="grid">{"".join(fig)}</div>{JS}</body></html>')
    (out_dir / f"{genus}.html").write_text(doc, encoding="utf-8")


def index_page(out_dir, df, thumb):
    g = df.groupby("genus")
    stats = g["status"].apply(lambda s: (s != "keep").sum())
    totals = g.size()
    first = g["path"].first()
    order = (totals - stats).sort_values(ascending=False)  # by keep count
    cards = []
    for genus in order.index:
        tot = int(totals[genus]); rm = int(stats[genus]); kp = tot - rm
        src = rel_src(out_dir, first[genus])
        cards.append(
            f'<a class="gcard" href="{quote(genus)}.html">'
            f'<img loading="lazy" src="{src}" alt="">'
            f'<div class="gname">{html.escape(genus)}</div>'
            f'<div class="gmeta">keep <b>{kp:,}</b> · remove <b style="color:#c0392b">{rm:,}</b></div>'
            f'</a>')
    n = len(df); nrm = int((df.status != "keep").sum()); nkp = n - nrm
    css = CSS.replace("THUMB", str(thumb)) + """
<style>.grid{grid-template-columns:repeat(auto-fill,minmax(160px,1fr))}
a.gcard{display:block;background:#fff;border-radius:8px;overflow:hidden;text-decoration:none;
  color:var(--ink);box-shadow:0 1px 3px rgba(0,0,0,.12)}
a.gcard img{width:100%;height:120px;object-fit:cover;display:block;background:#ddd}
.gname{font-weight:700;padding:6px 8px 0}.gmeta{font-size:12px;color:#555;padding:0 8px 8px}</style>"""
    head = (f'<header><h1>Curation review (2ndrev + flagged)</h1>'
            f'<span class="meta">{n:,} live → keep <b>{nkp:,}</b> · '
            f'remove <b>{nrm:,}</b> — click a genus</span></header>')
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<title>Curation review — index</title>{css}</head><body>'
           f'{head}<div class="grid">{"".join(cards)}</div></body></html>')
    (out_dir / "index.html").write_text(doc, encoding="utf-8")


def main():
    args = get_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    df, n_flag = load(args)
    nrm = int((df.status != "keep").sum())
    print(f"Live: {len(df):,} | flagged file entries: {n_flag:,}")
    print(f"Status counts: {dict(df.status.value_counts())}")
    print(f"KEEP {len(df)-nrm:,}  |  REMOVE {nrm:,}")
    for genus, rows in df.groupby("genus"):
        genus_page(out_dir, genus, rows.reset_index(drop=True), args.thumb)
    index_page(out_dir, df, args.thumb)
    print(f"Done -> open {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
