"""
Bake-off Orchestrator — ONG Orchid Identifier v3  (Colab)
=========================================================
Train + evaluate ALL candidate backbones sequentially on the IDENTICAL protocol,
then print a single comparison table. One file so the whole bake-off is one command
and is reproducible for the Methods Paper.

For each model key it does:
    1. train   →  03_train_bakeoff_colab.py  (saves models/<key>/best_model.pth + vocab.json)
    2. evaluate →  13_evaluate.py             (saves eval/<key>/results.json on test_live)
Then it reads every eval/<key>/results.json and prints the comparison vs the baseline.

Run from a Colab cell (after the setup cell — see below):
    ROOT = "/content/drive/MyDrive/orchid_project"
    %run {ROOT}/scripts/run_bakeoff_all_colab.py --root {ROOT} --photos /content/photos

Options:
    --models bioclip2 dinov2l           # restrict / reorder (default: all four, predicted-best first)
    --t4-safe                           # lower img/bs for the heavy ViT-L/ConvNeXt-L so they fit a T4
    --skip-train                        # only (re)evaluate models already trained
    --skip-eval                         # only train (fill the table later)

Eval recipe per key MUST mirror the REGISTRY in 03_train_bakeoff_colab.py (img sizes!).
"""

import argparse, json, shlex, subprocess, sys, time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")   # allow unicode prints on Windows cp1252
except Exception:
    pass

# eval recipe per trained model-key — keep img sizes in sync with the trainer REGISTRY
EVAL = {
    "bioclip2":    dict(arch="openclip", model="hf-hub:imageomics/bioclip-2",           img=224),
    "dinov2l":     dict(arch="timm",     model="vit_large_patch14_reg4_dinov2.lvd142m", img=448),
    "convnextv2l": dict(arch="timm",     model="convnextv2_large.fcmae_ft_in22k_in1k",  img=384),
    "effnetv2l":   dict(arch="timm",     model="tf_efficientnetv2_l.in21k_ft_in1k",     img=448),
}
ORDER = ["bioclip2", "dinov2l", "convnextv2l", "effnetv2l"]   # predicted-best / cheapest first

# FROZEN deterministic baseline (clean+merged split, 16,701 img / 120 genera) — the model
# to beat. OPTIMISTIC (train/test overlap) → even matching it is an improvement. macro≥5=76.2.
BASELINE = dict(macro_top1=74.0, macro_top5=84.8, global_top1=90.1, sp_r5=73.4, gn_r5=94.1)


def get_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--root", default="/content/drive/MyDrive/orchid_project",
                    help="Drive project root (holds data/splits, models/, eval/)")
    ap.add_argument("--photos", default="/content/photos",
                    help="local unzipped photos root (/content/photos/{Genus}/*.jpg)")
    ap.add_argument("--models", nargs="+", default=ORDER, choices=ORDER,
                    help="which backbones to run, in order")
    ap.add_argument("--train-script", default=None)
    ap.add_argument("--eval-script",  default=None)
    ap.add_argument("--t4-safe", action="store_true",
                    help="lower img/bs for heavy models so they fit a T4 (eval img follows)")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval",  action="store_true")
    ap.add_argument("--train-extra", default="",
                    help="extra args appended to EVERY training command, e.g. "
                         "\"--sampler-power 0 --loss ce\" (the corrected prior-respecting recipe)")
    return ap.parse_args()


def find_script(explicit, root, name):
    """Locate a script: explicit path, else scripts/ then notebooks/ under root."""
    if explicit:
        return Path(explicit)
    for sub in ("scripts", "notebooks"):
        p = Path(root) / sub / name
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} not found under {root}/scripts or {root}/notebooks "
                            f"— upload it or pass an explicit path.")


def run(cmd):
    """Run a subprocess, streaming its output live; return True on success."""
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd]).returncode == 0


def train_one(key, args, train_script):
    cmd = [sys.executable, train_script, "--model", key,
           "--drive-root", args.root, "--photos-root", args.photos]
    if args.t4_safe and key != "bioclip2":           # heavy models → shrink to fit T4
        cmd += ["--img-size", "224", "--batch-size", "32"]
    if args.train_extra:                              # corrected recipe, e.g. --sampler-power 0 --loss ce
        cmd += shlex.split(args.train_extra)
    return run(cmd)


def eval_one(key, args, eval_script):
    spec = EVAL[key]
    img = 224 if (args.t4_safe and key != "bioclip2") else spec["img"]   # MUST match training
    mdir = Path(args.root) / "models" / key
    # Evaluate BOTH selection checkpoints; fall back to legacy single best_model.pth.
    variants = [(tag, mdir / fn) for tag, fn in
                (("macro", "best_model_macro.pth"), ("global", "best_model_global.pth"))
                if (mdir / fn).exists()]
    if not variants:
        if (mdir / "best_model.pth").exists():
            variants = [("macro", mdir / "best_model.pth")]   # legacy run (one checkpoint)
        else:
            print(f"  [skip eval] no checkpoint in {mdir} — train it first."); return False
    ok = True
    for tag, ckpt in variants:
        out = Path(args.root) / "eval" / (key if tag == "macro" else f"{key}_{tag}")
        print(f"  -- eval {key} [{tag}] -> eval/{out.name}")
        cmd = [sys.executable, eval_script, "--arch", spec["arch"], "--model", spec["model"],
               "--img-size", str(img),
               "--checkpoint", ckpt, "--vocab", mdir / "vocab.json",
               "--splits-dir", Path(args.root) / "data" / "splits",
               "--photos-root", args.photos,
               "--out-dir", out]
        ok = run(cmd) and ok
    return ok


def read_row(label, root):
    f = Path(root) / "eval" / label / "results.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    c, q = r.get("classification", {}) or {}, r.get("retrieval", {}) or {}
    g = lambda d, k: (d.get(k) * 100) if d.get(k) is not None else None
    return dict(macro_top1=g(c, "macro_top1"), macro_top5=g(c, "macro_top5"),
                global_top1=g(c, "global_top1"),
                sp_r5=g(q, "species_recall@5"), gn_r5=g(q, "genus_recall@5"))


def fmt(x): return "—" if x is None else f"{x:5.1f}"


def print_table(keys, root):
    print("\n" + "=" * 82)
    print("BAKE-OFF COMPARISON  (test_live, identical protocol — macro top-1 AND global top-1)")
    print("=" * 82)
    head = f"{'model':18s} {'macroT1':>8s} {'macroT5':>8s} {'globalT1':>9s} {'spR@5':>7s} {'gnR@5':>7s}"
    print(head); print("-" * len(head))
    print(f"{'baseline effb4':18s} {fmt(BASELINE['macro_top1']):>8s} {fmt(BASELINE['macro_top5']):>8s} "
          f"{fmt(BASELINE['global_top1']):>9s} {fmt(BASELINE['sp_r5']):>7s} {fmt(BASELINE['gn_r5']):>7s}")
    # each key may have a macro-ckpt (eval/<key>) and a global-ckpt (eval/<key>_global)
    labels = []
    for k in keys:
        labels.append(k)
        if (Path(root) / "eval" / f"{k}_global").exists():
            labels.append(f"{k}_global")
    best_cls, best_ret = None, None
    for lab in labels:
        r = read_row(lab, root)
        if r is None:
            print(f"{lab:18s} {'(no results.json — not evaluated yet)':>48s}"); continue
        print(f"{lab:18s} {fmt(r['macro_top1']):>8s} {fmt(r['macro_top5']):>8s} "
              f"{fmt(r['global_top1']):>9s} {fmt(r['sp_r5']):>7s} {fmt(r['gn_r5']):>7s}")
        if r["macro_top1"] is not None and (best_cls is None or r["macro_top1"] > best_cls[1]):
            best_cls = (lab, r["macro_top1"])
        if r["sp_r5"] is not None and (best_ret is None or r["sp_r5"] > best_ret[1]):
            best_ret = (lab, r["sp_r5"])
    print("-" * len(head))
    if best_cls:
        delta = best_cls[1] - BASELINE["macro_top1"]
        print(f"WINNER genus (macro top-1): {best_cls[0]}  {best_cls[1]:.1f}%  "
              f"({'+' if delta>=0 else ''}{delta:.1f} vs baseline)")
    if best_ret:
        print(f"WINNER retrieval (species R@5): {best_ret[0]}  {best_ret[1]:.1f}%")
    print("Record these in ONGv3_progress.Rmd → fill the Tahap A table in ONGv3_plan.Rmd.")


def main():
    args = get_args()
    train_script = find_script(args.train_script, args.root, "03_train_bakeoff_colab.py")
    eval_script  = find_script(args.eval_script,  args.root, "13_evaluate.py")
    print(f"Root: {args.root}\nTrain: {train_script}\nEval:  {eval_script}")
    print(f"Models (in order): {args.models}"
          f"{'  [T4-safe: heavy models @224/bs32]' if args.t4_safe else ''}")

    for key in args.models:
        t0 = time.time()
        print(f"\n{'#'*78}\n# {key}\n{'#'*78}")
        if not args.skip_train:
            if not train_one(key, args, train_script):
                print(f"  !! training FAILED for {key} — continuing to next model."); continue
        if not args.skip_eval:
            if not eval_one(key, args, eval_script):
                print(f"  !! eval skipped/failed for {key}.")
        print(f"  ({key} done in {(time.time()-t0)/60:.1f} min)")

    print_table(args.models, args.root)


if __name__ == "__main__":
    main()
