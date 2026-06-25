"""
ONG Orchid Identifier v3 — Hugging Face Space
Two-stage: DINOv2 ViT-L/14 genus classifier + FAISS species retrieval
Top-5 genera | Top-5 species x up to 5 similar photos

Differences vs v2 (EfficientNet-B4):
  - backbone = timm vit_large_patch14_reg4_dinov2.lvd142m @448 (must pass img_size)
  - embedding = forward_head(pre_logits=True) (1024-d), NOT the classifier=Identity trick
  - mean/std resolved from the model config (DINOv2 = ImageNet norm)
"""

import io
import json
import os
import requests
import numpy as np
import torch
import torch.nn.functional as F
import timm
import faiss
import gradio as gr
from huggingface_hub import hf_hub_download
from torchvision import transforms
from datetime import datetime
from PIL import Image
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "vit_large_patch14_reg4_dinov2.lvd142m"
IMG_SIZE        = 448
MODEL_DIR       = Path("models/dinov2l")
# Weights (608 MB fp16) live in a separate public model repo — the Space free tier caps
# repo storage at 1 GB, so the checkpoint is fetched at startup instead of bundled.
MODEL_REPO      = "Rezamonium/ong-dinov2l-v3"
CKPT_FILE       = "best_model.pth"
FAISS_PATH      = Path("models/ong_species_index.faiss")
META_PATH       = Path("models/ong_metadata.json")
WPA_COUNTS_PATH = Path("models/genus_wpa_counts.json")
THUMB_DIR       = Path("thumbnails")
HF_DATASET_REPO = "Rezamonium/birdshead-community-photos"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Calibration & open-set gate ───────────────────────────────────────────────
# Temperature scaling: divide logits by T before softmax (Guo et al., 2017). The DINOv2
# classifier is under-confident; T<1 sharpens it (ECE 0.178 -> 0.029, accuracy unchanged).
# Fit offline on val_live.csv (notebooks/27_reliability_diagram.py).
TEMPERATURE = 0.728
# Open-set novelty gate: novelty = 1 - cosine sim to the nearest of ALL reference embeddings.
# tau recalibrated on the deployed 120-genus bank (notebooks/28_deploy_openset_tau.py).
TAU_05    = 0.3108   # Conservative (default): ~5% false-positive rate on known genera
TAU_10    = 0.2333   # Sensitive: ~10% FPR, flags more probable novel genera
CONF_GATE = 0.50     # genus softmax (post-temperature) below this = ambiguous among known genera

# ── Load model ────────────────────────────────────────────────────────────────
with open(MODEL_DIR / "vocab.json") as f:
    genera = json.load(f)
n_cls = len(genera)

model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=n_cls, img_size=IMG_SIZE)
cfg = timm.data.resolve_model_data_config(model)
MEAN, STD = list(cfg["mean"]), list(cfg["std"])

_ckpt_path = hf_hub_download(repo_id=MODEL_REPO, filename=CKPT_FILE)
_state = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
_state = _state.get("state_dict", _state) if isinstance(_state, dict) else _state
# fp16 weights load into the fp32 model (load_state_dict casts) → CPU inference stays fp32.
missing, unexpected = model.load_state_dict(_state, strict=False)
if missing or unexpected:
    print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
model = model.to(DEVICE).eval()

# ── Load FAISS + metadata ─────────────────────────────────────────────────────
faiss_index = faiss.read_index(str(FAISS_PATH))
with open(META_PATH, encoding="utf-8") as f:
    metadata = json.load(f)

genus_to_idx = defaultdict(list)
for i, rec in enumerate(metadata):
    genus_to_idx[rec["genus"]].append(i)

with open(WPA_COUNTS_PATH, encoding="utf-8") as f:
    genus_wpa_counts = json.load(f)

print(f"Model: {n_cls} genera ({MODEL_NAME}) | FAISS: {faiss_index.ntotal:,} vectors | Device: {DEVICE}")

# ── Wikipedia summary cache ───────────────────────────────────────────────────
_wiki_cache: dict = {}

def get_wiki_summary(genus: str) -> str:
    if genus in _wiki_cache:
        return _wiki_cache[genus]
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{genus}"
        r = requests.get(url, timeout=4, headers={"User-Agent": "ONG-Orchid-Identifier/3.0"})
        if r.status_code == 200:
            extract = r.json().get("extract", "")
            sentences = extract.split(". ")
            summary = ". ".join(sentences[:2]).strip()
            if summary and not summary.endswith("."):
                summary += "."
            _wiki_cache[genus] = summary
            return summary
    except Exception:
        pass
    _wiki_cache[genus] = ""
    return ""


def get_genus_info(genus: str, conf: float) -> str:
    if conf >= 0.80:
        conf_label = "High confidence"
    elif conf >= 0.50:
        conf_label = "Medium confidence"
    else:
        conf_label = "Low confidence"
    wiki = get_wiki_summary(genus)
    wpa_count = genus_wpa_counts.get(genus, 0)
    wpa_line = f"In West Papua, {wpa_count} species have been recorded." if wpa_count else ""
    parts = [f"**{conf_label}: {genus}**"]
    if wiki:
        parts.append(wiki)
    if wpa_line:
        parts.append(wpa_line)
    parts.append(
        "[Explore in the database →](https://birdsheadorchid.id/identify/key/key/Media/Html/index.htm)"
        "  ·  "
        "[Verify with Lucid identification key →](https://birdsheadorchid.id/identify/key_player.html)"
    )
    return "\n\n".join(parts)

# ── Image transform (same as 13_evaluate.py) ──────────────────────────────────
tfm = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.1)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

PLACEHOLDER = Image.new("RGB", (300, 300), (220, 230, 220))


def load_thumbnail(thumb_path: str) -> Image.Image:
    p = Path(thumb_path)
    if thumb_path and p.exists():
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            pass
    return PLACEHOLDER


@torch.no_grad()
def extract_features(img: Image.Image):
    """Single-image forward — returns (probs np.array, embedding np.array 1024-d)."""
    tensor = tfm(img).unsqueeze(0).to(DEVICE)
    with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
        feat   = model.forward_features(tensor)
        emb    = model.forward_head(feat, pre_logits=True)
        logits = model.get_classifier()(emb)
    probs = F.softmax(logits.float() / TEMPERATURE, dim=1)[0].cpu().numpy().astype("float32")
    emb   = emb.float().cpu().numpy().astype("float32")
    return probs, emb


def identify(imgs: list, top_n: int = 5, photos_per_species: int = 5):
    """Multi-photo late fusion: weighted-average genus probs + embeddings, then FAISS."""
    all_probs, all_embs = [], []
    for img in imgs:
        probs, emb = extract_features(img)
        all_probs.append(probs)
        all_embs.append(emb)

    weights = np.array([p.max() for p in all_probs], dtype="float32")
    weights /= weights.sum()
    avg_probs = np.average(all_probs, axis=0, weights=weights)
    avg_emb   = np.average(all_embs,  axis=0, weights=weights).reshape(1, -1).astype("float32")
    faiss.normalize_L2(avg_emb)

    # Open-set novelty: cosine distance to the nearest of ALL references (global search).
    nov_D, _ = faiss_index.search(avg_emb, 1)
    novelty  = 1.0 - float(nov_D[0, 0])

    top5_i = avg_probs.argsort()[::-1][:5]
    top5_p = avg_probs[top5_i]
    top3_i, top3_p = top5_i[:3], top5_p[:3]
    genus_labels = {genera[i]: float(p) for i, p in zip(top5_i, top5_p)}

    species_pool = defaultdict(list)
    for g_prob, g_idx in zip(top3_p.tolist(), top3_i.tolist()):
        genus     = genera[g_idx]
        g_indices = genus_to_idx.get(genus, [])
        if not g_indices:
            continue
        g_vecs  = np.array([faiss_index.reconstruct(i) for i in g_indices], dtype="float32")
        sub_idx = faiss.IndexFlatIP(g_vecs.shape[1])
        sub_idx.add(g_vecs)
        k       = min(50, len(g_indices))
        D, I    = sub_idx.search(avg_emb, k)
        for score, local_i in zip(D[0], I[0]):
            rec      = metadata[g_indices[local_i]]
            combined = g_prob * float(score)
            species_pool[rec["species"]].append({
                "genus":      genus,
                "genus_conf": g_prob,
                "score":      combined,
                "path":       rec.get("thumb_path", ""),
            })

    for sp in species_pool:
        species_pool[sp].sort(key=lambda x: x["score"], reverse=True)
        species_pool[sp] = species_pool[sp][:photos_per_species]

    ranked = sorted(species_pool.items(), key=lambda kv: kv[1][0]["score"], reverse=True)[:top_n]
    results = [{"species": sp, "genus": ph[0]["genus"], "genus_conf": ph[0]["genus_conf"],
                "best_score": ph[0]["score"], "photos": ph} for sp, ph in ranked]
    return results, genus_labels, novelty


def _safe_name(text: str, maxlen: int = 80) -> str:
    import re
    s = re.sub(r"[^\w.\-]+", "", (text or "").strip().replace(" ", "_"))
    return s[:maxlen]


def save_to_hf_dataset(image, genus, species, genus_conf, timestamp, photo_idx=1, contributor=""):
    try:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            return False, "HF_TOKEN secret not configured."
        from huggingface_hub import HfApi
        api = HfApi()
        conf_pct     = int(genus_conf * 100)
        safe_species = species.replace(" ", "_")
        contrib      = _safe_name(contributor)
        contrib_part = f"{contrib}_" if contrib else ""
        filename     = f"{genus}/{timestamp}_{contrib_part}{safe_species}_{conf_pct}pct_p{photo_idx}.jpg"
        img_bytes    = io.BytesIO()
        image.save(img_bytes, format="JPEG", quality=90)
        img_bytes.seek(0)
        api.upload_file(path_or_fileobj=img_bytes, path_in_repo=filename,
                        repo_id=HF_DATASET_REPO, repo_type="dataset", token=hf_token)
        return True, filename
    except Exception as e:
        return False, str(e)


def _load_gallery_imgs(gallery_input):
    imgs = []
    if not gallery_input:
        return imgs
    for item in list(gallery_input):
        try:
            src = item[0] if isinstance(item, (list, tuple)) else item
            if isinstance(src, str):
                imgs.append(Image.open(src).convert("RGB"))
            elif isinstance(src, Image.Image):
                imgs.append(src.convert("RGB"))
            elif isinstance(src, np.ndarray):
                imgs.append(Image.fromarray(src).convert("RGB"))
            elif hasattr(src, "name"):
                imgs.append(Image.open(src.name).convert("RGB"))
        except Exception:
            pass
    return imgs


def predict(gallery_input, consent, contributor="", sensitivity="Conservative (fewer warnings)"):
    all_imgs = _load_gallery_imgs(gallery_input)
    if not all_imgs:
        return {}, "", [], "Upload at least one photo to begin.", ""

    imgs = all_imgs[:3]
    results, genus_labels, novelty = identify(imgs)
    n_photos = len(imgs)

    best = results[0] if results else None
    genus_info = get_genus_info(best["genus"], best["genus_conf"]) if best else ""

    # Soft open-set gate: never hide results, just prepend a verification nudge when the query
    # looks like an unseen genus (novelty > tau) and/or the genus prediction is weak.
    tau          = TAU_10 if sensitivity.startswith("Sensitive") else TAU_05
    is_novel     = novelty > tau
    is_ambiguous = bool(best) and best["genus_conf"] < CONF_GATE
    if best and (is_novel or is_ambiguous):
        if is_novel and is_ambiguous:
            reason = ("This specimen looks **unlike any genus the model was trained on**, and the "
                      "genus prediction is weak.")
        elif is_novel:
            reason = ("This specimen looks **unlike any genus the model was trained on** — it may "
                      "be a genus outside the 120 the model knows.")
        else:
            reason = "The genus prediction is **weak** — several known genera score similarly."
        banner = ("> ⚠️ **Verify this identification.** " + reason + " Treat the matches below as "
                  "suggestions and confirm with morphology or the "
                  "[Lucid multi-access key](https://birdsheadorchid.id/identify/key_player.html).")
        genus_info = banner + "\n\n" + genus_info

    gallery = []
    for rank, res in enumerate(results, 1):
        for j, photo in enumerate(res["photos"], 1):
            caption = (f"#{rank} {res['species']}  [{j}/{len(res['photos'])}]\n"
                       f"Genus: {res['genus']} ({res['genus_conf']*100:.0f}%)  |  "
                       f"Score: {photo['score']:.3f}")
            gallery.append((load_thumbnail(photo["path"]), caption))

    photo_note = f" · {n_photos} photos fused" if n_photos > 1 else ""
    summary = (
        f"<div style='text-align:center; padding:8px'>"
        f"<p style='font-size:1.5em; font-weight:bold; margin:0'>{best['species']}</p>"
        f"<p style='margin:4px 0 0 0; color:#555'>Genus: {best['genus']} · "
        f"confidence {best['genus_conf']*100:.0f}%{photo_note} · score {best['best_score']:.3f}</p>"
        f"</div>"
        if best else "No results."
    )

    storage_msg = ""
    if consent and best:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved, failed, last_err = 0, 0, ""
        for idx, img in enumerate(all_imgs, 1):
            ok, result = save_to_hf_dataset(img, best["genus"], best["species"],
                                            best["genus_conf"], timestamp=timestamp,
                                            photo_idx=idx, contributor=contributor)
            if ok:
                saved += 1
            else:
                failed += 1; last_err = result
        if saved:
            storage_msg = f"✅ {saved} photo(s) saved to research dataset. Thank you for contributing!"
            if failed:
                storage_msg += f" ({failed} failed: {last_err})"
        else:
            storage_msg = f"⚠️ Could not save photos: {last_err}"
    elif not consent:
        storage_msg = "Photo not saved — consent not given."

    return genus_labels, genus_info, gallery, summary, storage_msg


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="New Guinea Orchid Identifier v3", theme=gr.themes.Default()) as demo:
    gr.Markdown(
        """
        # 🌿 New Guinea Orchid Identifier v3
        **Orchids of New Guinea — Genus & Species Classifier**

        Upload 1–3 photos of the same orchid (different angles) to identify its genus. More photos = better accuracy. For species-level identification, use the [Lucid multi-access identification key](https://birdsheadorchid.id/identify/key_player.html).

        The AI runs in two stages:
        - **Stage 1** — DINOv2 ViT-L/14 predicts the top-5 *genera* (trained on 11,677 live field photos · 120 genera)
        - **Stage 2** — FAISS visual similarity retrieves the top-5 most similar *species* with matching photos

        *Genus recall@5: 99% · species recall@5: 87% (single photo)*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Gallery(
                label="Upload orchid photos (1–3, same specimen, different angles)",
                columns=3, rows=1, height=200, interactive=True,
                object_fit="cover", show_label=True,
            )
            consent_chk = gr.Checkbox(
                value=True,
                label=("I consent to this photo being saved to the Bird's Head orchid "
                       "research dataset to help improve future identification accuracy."),
            )
            contributor_txt = gr.Textbox(
                label="Your name & photo location (optional)",
                placeholder="e.g. Reza_Sorong",
                info="Saved into the dataset filename when you consent, so your contribution is credited.",
            )
            sensitivity = gr.Radio(
                ["Conservative (fewer warnings)", "Sensitive (more warnings)"],
                value="Conservative (fewer warnings)",
                label="Novel-genus warning sensitivity",
                info="How readily to flag photos that may be a genus the model has not seen.",
            )
            run_btn     = gr.Button("Identify", variant="primary")
            summary_out = gr.Markdown(label="Best Match")
            storage_out = gr.Markdown()
        with gr.Column(scale=2):
            genus_out      = gr.Label(num_top_classes=5, label="Stage 1 — Genus Prediction (top 5)")
            genus_info_out = gr.Markdown()

    gallery_out = gr.Gallery(
        label="Stage 2 — Top-5 Species × 5 Similar Photos",
        columns=5, rows=5, height=650, object_fit="cover"
    )

    gr.Markdown(
        "**Privacy notice:** Photos are only saved if you tick the consent box above. "
        "Saved photos are used solely for improving the Bird's Head orchid identification model "
        "and may be shared as an open scientific dataset. No personal information is collected."
    )

    gr.Markdown(
        """**About this tool**

This AI model is currently under development, built as part of the Bird's Head Peninsula Orchid Identification Key (Lucid) project.

The classifier recognises **120 genera** of orchids from New Guinea. Model: DINOv2 ViT-L/14 fine-tuned on 11,677 live field photographs.

⚠️ This AI prediction is a probabilistic suggestion and should be verified using morphological characters or the [Lucid identification key](https://birdsheadorchid.id/identify/key_player.html). Low confidence scores (<70%) indicate the image may not contain a recognisable orchid or the genus is underrepresented in training data."""
    )

    gr.Markdown("_Part of the [Orchids of the Bird's Head Peninsula](https://birdsheadorchid.id) project._")

    run_btn.click(fn=predict, inputs=[img_input, consent_chk, contributor_txt, sensitivity],
                  outputs=[genus_out, genus_info_out, gallery_out, summary_out, storage_out])

demo.launch()
