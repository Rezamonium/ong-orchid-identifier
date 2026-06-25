"""
17_compare_models.py — Build the ViT-vs-CNN comparison tables for the ONG v3 paper.

Reads, for each of the 4 clean pilots, from <eval-dir>/<key>_global/:
  - results.json                         (TEST-set eval: top-1/5, macro, ECE, CIs, retrieval R@k)
  - per_genus_accuracy.csv               (per-genus support / top1 / top5 on TEST)
  - "*folder stats*/results.json"        (training recipe + best VAL macro/top1)  [optional]

Emits to <eval-dir>/_comparison/:
  - comparison_main.csv   full numeric table (all metrics + 95% CIs, full precision)
  - comparison_main.md    paper-ready summary table (point estimates, best per column bolded)
  - comparison_ci.md      key metrics with 95% bootstrap CIs
  - per_genus.csv         genus x model top-1 accuracy (+ support), wide format
  - per_genus.md          same, Markdown (sorted by support desc) for the supplement

Stdlib only — no pandas/numpy. Run locally:
  python notebooks/17_compare_models.py
  python notebooks/17_compare_models.py --eval-dir "E:/Claude Code/ONG_v3/colab/result/eval"
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Windows consoles default to cp1252 and choke on →, ·, em-dash. Files are UTF-8 regardless.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# key -> (display name, architecture family). Order here is the canonical paper order.
MODELS = {
    "dinov2l":     ("DINOv2 ViT-L/14",    "ViT"),
    "bioclip2":    ("BioCLIP-2 ViT-L/14", "ViT"),
    "convnextv2l": ("ConvNeXt-V2-L",      "CNN"),
    "effnetv2l":   ("EfficientNetV2-L",   "CNN"),
}

DEFAULT_EVAL_DIR = Path(r"E:/Claude Code/ONG_v3/colab/result/eval")

TOP_CONFUSIONS = 10  # confused pairs to show per model in confusions.md


# ── loading ─────────────────────────────────────────────────────────────────────
def load_model(key: str, eval_dir: Path) -> dict | None:
    """Collect every metric we report for one model into a flat dict."""
    mdir = eval_dir / f"{key}_global"
    res_path = mdir / "results.json"
    if not res_path.exists():
        print(f"  [skip] {key}: no results.json at {res_path}")
        return None

    with open(res_path, encoding="utf-8") as f:
        r = json.load(f)
    cls, ret = r.get("classification", {}), r.get("retrieval", {})

    name, family = MODELS[key]
    row = {
        "key": key, "model": name, "family": family,
        "backbone": r.get("model", ""), "img_size": r.get("img_size", ""),
        # classification (TEST)
        "n_test": cls.get("n_test"), "n_genera": cls.get("n_genera_evaluated"),
        "macro_top1": pct(cls.get("macro_top1")), "macro_top5": pct(cls.get("macro_top5")),
        "global_top1": pct(cls.get("global_top1")), "global_top5": pct(cls.get("global_top5")),
        "ece": cls.get("ece"),
        "macro_top1_ci": ci(cls.get("macro_top1_ci95")),
        "macro_top5_ci": ci(cls.get("macro_top5_ci95")),
        "global_top1_ci": ci(cls.get("global_top1_ci95")),
        # retrieval (TEST)
        "embed_dim": ret.get("embed_dim"),
        "species_r1": pct(ret.get("species_recall@1")), "species_r5": pct(ret.get("species_recall@5")),
        "species_r10": pct(ret.get("species_recall@10")),
        "genus_r1": pct(ret.get("genus_recall@1")), "genus_r5": pct(ret.get("genus_recall@5")),
        "species_r5_ci": ci(ret.get("species_recall@5_ci95")),
        "genus_r5_ci": ci(ret.get("genus_recall@5_ci95")),
    }

    # training recipe + best VAL metrics (provenance for the methods section)
    stats = next(mdir.glob("*folder stats*/results.json"), None)
    if stats:
        with open(stats, encoding="utf-8") as f:
            s = json.load(f)
        row.update({
            "select": s.get("select"), "loss": s.get("loss"),
            "sampler_power": s.get("sampler_power"),
            "best_val_macro": s.get("best_val_macro"), "best_val_top1": s.get("best_val_top1"),
        })
    return row


def load_per_genus(key: str, eval_dir: Path) -> dict:
    """genus -> (support, top1_acc%) for one model."""
    p = eval_dir / f"{key}_global" / "per_genus_accuracy.csv"
    out = {}
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["genus"]] = (int(r["support"]), pct(float(r["top1_acc"])))
    return out


def load_confused(key: str, eval_dir: Path) -> list[dict]:
    """List of {true, pred, count, frac%} confused pairs for one model (count desc)."""
    p = eval_dir / f"{key}_global" / "confused_pairs.csv"
    rows = []
    if not p.exists():
        return rows
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"true": r["true"], "pred": r["pred"],
                         "count": int(r["count"]), "frac": pct(float(r["frac_of_true"]))})
    rows.sort(key=lambda x: x["count"], reverse=True)
    return rows


# ── formatting helpers ──────────────────────────────────────────────────────────
def pct(x):
    return None if x is None else round(float(x) * 100, 1)


def ci(pair):
    if not pair:
        return ""
    lo, hi = pair
    return f"[{lo*100:.1f}, {hi*100:.1f}]"


def fmt(v):
    return "" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))


def md_table(headers, rows, bold_best=None):
    """bold_best: {col_index: 'max'|'min'} — bold the winning cell in those columns."""
    rows = [list(map(fmt, r)) for r in rows]
    if bold_best:
        for ci_, mode in bold_best.items():
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[ci_]))
                except (ValueError, IndexError):
                    vals.append(None)
            present = [v for v in vals if v is not None]
            if not present:
                continue
            if len(set(present)) <= 1:
                continue  # all tied → no meaningful winner, don't bold
            best = max(present) if mode == "max" else min(present)
            for r, v in zip(rows, vals):
                if v is not None and abs(v - best) < 1e-9:
                    r[ci_] = f"**{r[ci_]}**"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# ── main ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    ap.add_argument("--hardest-min-support", type=int, default=10,
                    help="min test support for a genus to count in the hardest-10 table")
    args = ap.parse_args()
    eval_dir = args.eval_dir
    out_dir = eval_dir / "_comparison"
    out_dir.mkdir(exist_ok=True)

    print(f"Eval dir: {eval_dir}")
    models = [m for m in (load_model(k, eval_dir) for k in MODELS) if m]
    if not models:
        print("No models found — check --eval-dir."); return
    # rank by genus macro top-1 (the headline metric)
    models.sort(key=lambda m: m["macro_top1"] or 0, reverse=True)
    print(f"Loaded {len(models)} models: {', '.join(m['key'] for m in models)}")

    n_test = next((m["n_test"] for m in models if m["n_test"]), "?")
    n_gen  = next((m["n_genera"] for m in models if m["n_genera"]), "?")

    # ── full numeric CSV ──────────────────────────────────────────────────────────
    cols = ["model", "family", "backbone", "img_size", "n_test", "n_genera",
            "macro_top1", "macro_top1_ci", "macro_top5", "macro_top5_ci",
            "global_top1", "global_top1_ci", "global_top5", "ece",
            "species_r1", "species_r5", "species_r5_ci", "species_r10",
            "genus_r1", "genus_r5", "genus_r5_ci", "embed_dim",
            "best_val_macro", "best_val_top1", "select", "loss", "sampler_power"]
    with open(out_dir / "comparison_main.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(models)

    # ── paper-ready main MD (bold best per column) ────────────────────────────────
    headers = ["Model", "Family", "Res", "macro T1", "macro T5",
               "global T1", "global T5", "species R@5", "genus R@5", "ECE", "val macro"]
    rows = [[m["model"], m["family"], m["img_size"], m["macro_top1"], m["macro_top5"],
             m["global_top1"], m["global_top5"], m["species_r5"], m["genus_r5"],
             f"{m['ece']:.3f}" if m["ece"] is not None else "", m.get("best_val_macro")]
            for m in models]
    bold = {3: "max", 4: "max", 5: "max", 6: "max", 7: "max", 8: "max", 9: "min"}
    main_md = (
        f"# ONG v3 — Model comparison (ViT vs CNN)\n\n"
        f"Test set: **{n_test} images · {n_gen} genera evaluated** · "
        f"bootstrap n=1000 · checkpoint selected by val global top-1. "
        f"All values are percentages; **bold** = best in column (ECE: lower is better).\n\n"
        + md_table(headers, rows, bold) + "\n\n"
        "Res = input resolution (px). macro = macro-averaged over genera (= balanced accuracy). "
        "species/genus R@5 = FAISS retrieval recall@5. ECE = expected calibration error. "
        "val macro = best validation macro-F1 during training (provenance).\n\n"
        "> Confound: backbones run at their native resolutions (ViT-DINOv2/EffNetV2 448, "
        "ConvNeXt 384, BioCLIP 224); read as best deployable config per backbone, not a "
        "resolution-matched test.\n"
    )
    (out_dir / "comparison_main.md").write_text(main_md, encoding="utf-8")

    # ── CI MD ─────────────────────────────────────────────────────────────────────
    ci_headers = ["Model", "macro T1 [95% CI]", "global T1 [95% CI]",
                  "species R@5 [95% CI]", "genus R@5 [95% CI]"]
    ci_rows = [[m["model"],
                f"{fmt(m['macro_top1'])} {m['macro_top1_ci']}",
                f"{fmt(m['global_top1'])} {m['global_top1_ci']}",
                f"{fmt(m['species_r5'])} {m['species_r5_ci']}",
                f"{fmt(m['genus_r5'])} {m['genus_r5_ci']}"] for m in models]
    (out_dir / "comparison_ci.md").write_text(
        "# ONG v3 — Key metrics with 95% bootstrap CIs\n\n"
        + md_table(ci_headers, ci_rows) + "\n", encoding="utf-8")

    # ── per-genus wide table ──────────────────────────────────────────────────────
    pg = {m["key"]: load_per_genus(m["key"], eval_dir) for m in models}
    all_genera = sorted(
        {g for d in pg.values() for g in d},
        key=lambda g: (-next((d[g][0] for d in pg.values() if g in d), 0), g),
    )
    keys = [m["key"] for m in models]
    names = [m["model"] for m in models]

    pg_csv_cols = ["genus", "support"] + [f"{k}_top1" for k in keys]
    with open(out_dir / "per_genus.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(pg_csv_cols)
        for g in all_genera:
            support = next((pg[k][g][0] for k in keys if g in pg[k]), "")
            w.writerow([g, support] + [pg[k].get(g, ("", ""))[1] for k in keys])

    pg_headers = ["Genus", "n"] + names
    pg_rows = []
    for g in all_genera:
        support = next((pg[k][g][0] for k in keys if g in pg[k]), "")
        pg_rows.append([g, support] + [pg[k].get(g, ("", ""))[1] for k in keys])
    pg_bold = {i: "max" for i in range(2, 2 + len(keys))}
    (out_dir / "per_genus.md").write_text(
        f"# ONG v3 — Per-genus top-1 accuracy (TEST, n={n_test})\n\n"
        f"{len(all_genera)} genera with test support, sorted by support desc. "
        "Values are top-1 accuracy (%); **bold** = best model for that genus.\n\n"
        + md_table(pg_headers, pg_rows, pg_bold) + "\n", encoding="utf-8")

    # ── hardest 10 genera (lowest mean top-1 across all models) ────────────────────
    def mean_top1(g):
        vals = [pg[k][g][1] for k in keys if g in pg[k] and pg[k][g][1] != ""]
        return sum(vals) / len(vals) if vals else 0.0

    def support(g):
        return next((pg[k][g][0] for k in keys if g in pg[k]), 0)

    eligible = [g for g in all_genera if support(g) >= args.hardest_min_support]
    hardest = sorted(eligible, key=lambda g: (mean_top1(g), -support(g)))[:10]
    hard_headers = ["Genus", "n", "Mean top-1"] + names
    hard_rows = []
    for g in hardest:
        hard_rows.append([g, support(g), round(mean_top1(g), 1)]
                         + [pg[k].get(g, ("", ""))[1] for k in keys])
    hard_bold = {i: "max" for i in range(3, 3 + len(keys))}
    (out_dir / "hardest_genera.md").write_text(
        f"# ONG v3 — 10 hardest genera (TEST, n={n_test})\n\n"
        f"Lowest mean top-1 accuracy across all 4 models, among genera with test "
        f"support n>={args.hardest_min_support} (a genus all backbones struggle on, "
        "excluding tiny-support sampling noise). Sorted by mean ascending, ties broken by "
        "larger support. Values are top-1 (%); **bold** = best model for that genus.\n\n"
        + md_table(hard_headers, hard_rows, hard_bold) + "\n", encoding="utf-8")

    # ── confusion analysis ────────────────────────────────────────────────────────
    conf = {m["key"]: load_confused(m["key"], eval_dir) for m in models}

    # long-format CSV: every confused pair from every model
    with open(out_dir / "confusions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "true", "pred", "count", "frac_of_true_pct"])
        for m in models:
            for c in conf[m["key"]]:
                w.writerow([m["model"], c["true"], c["pred"], c["count"], c["frac"]])

    # error sinks: predicted genera that absorb the most misclassifications, summed across models
    sinks_by_model = {m["key"]: {} for m in models}
    for m in models:
        for c in conf[m["key"]]:
            if c["true"] != c["pred"]:
                sinks_by_model[m["key"]][c["pred"]] = \
                    sinks_by_model[m["key"]].get(c["pred"], 0) + c["count"]
    totals = {}
    for d in sinks_by_model.values():
        for pred, n in d.items():
            totals[pred] = totals.get(pred, 0) + n
    top_sinks = sorted(totals, key=totals.get, reverse=True)[:6]
    sink_headers = ["Predicted genus (sink)", "Total"] + [m["model"] for m in models]
    sink_rows = [[pred, totals[pred]] + [sinks_by_model[m["key"]].get(pred, 0) for m in models]
                 for pred in top_sinks]

    # per-model top confused pairs
    parts = [
        f"# ONG v3 — Confusion analysis (TEST, n={n_test})\n",
        "Restricted to the top confused pairs each model logs. `count` = test images of the "
        "*true* genus predicted as the *pred* genus; `% of true` = share of that true genus's "
        "test images lost to this confusion.\n",
        "## Error sinks — genera that absorb the most misclassifications\n",
        "Sum of off-diagonal `count` landing on each predicted genus (across logged pairs), "
        "totalled over all 4 models. These attractor classes explain most long-tail failures.\n",
        md_table(sink_headers, sink_rows, {1: "max"}), "",
    ]
    for m in models:
        rows_c = [[c["true"], "→ " + c["pred"], c["count"], c["frac"]]
                  for c in conf[m["key"]][:TOP_CONFUSIONS]]
        parts.append(f"## {m['model']} — top {len(rows_c)} confused pairs\n")
        parts.append(md_table(["True genus", "Predicted", "n", "% of true"], rows_c))
        parts.append("")
    (out_dir / "confusions.md").write_text("\n".join(parts) + "\n", encoding="utf-8")

    # ── console preview ───────────────────────────────────────────────────────────
    print("\n" + main_md)
    print(f"Wrote → {out_dir}")
    for f in ["comparison_main.csv", "comparison_main.md", "comparison_ci.md",
              "per_genus.csv", "per_genus.md", "hardest_genera.md",
              "confusions.csv", "confusions.md"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
