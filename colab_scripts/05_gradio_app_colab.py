"""
ONG Orchid Identifier -- Gradio App (Colab Version)
Two-stage: EfficientNet-B4 genus classifier + FAISS species retrieval
Top-5 species x 5 most-similar photos = up to 25 photos in gallery.
"""
# -- Cell 11: Gradio App -- Top-5 Species x 5 Photos Gallery -----------------
# Gallery shows up to 25 photos (5 species x 5 most-similar photos each).
# Run AFTER Cell 8 (FAISS index built).
import torch, json, io
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import timm, faiss, gradio as gr
from torchvision import transforms
from PIL import Image
from pathlib import Path
from collections import defaultdict

EFFNET_DIR   = Path(f"{DRIVE_PROJECT}/models/efficientnet_b4")
DRIVE_MODELS = Path(f"{DRIVE_PROJECT}/models")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(EFFNET_DIR/"vocab.json") as f:
    genera = json.load(f)
n_cls = len(genera)

model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=n_cls)
model.load_state_dict(torch.load(EFFNET_DIR/"best_model.pth",
                                  map_location=DEVICE, weights_only=False))
model = model.to(DEVICE).eval()

faiss_index = faiss.read_index(str(DRIVE_MODELS/"ong_species_index.faiss"))
with open(DRIVE_MODELS/"ong_metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

genus_to_idx = defaultdict(list)
for i, rec in enumerate(metadata):
    genus_to_idx[rec["genus"]].append(i)

print(f"Model: {n_cls} genera | FAISS: {faiss_index.ntotal:,} vectors")

tfm = transforms.Compose([
    transforms.Resize(int(380*1.1)),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def remap(path):
    parts = path.replace("\\\\", "/").replace("\\", "/").split("/")
    try:
        pi = [x.lower() for x in parts].index("photos")
        return "/content/" + "/".join(parts[pi:])
    except ValueError:
        return path

def identify(img, top_n_species=5, photos_per_species=5):
    """
    Two-stage identification.
    Returns: (top5_species_results, genus_probs_top5)
    Each species result: {species, genus, best_score, genus_conf, photos: [list of {path, score}]}
    """
    tensor = tfm(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]
        top3_p, top3_i = probs.topk(3)
        top5_p, top5_i = probs.topk(5)
        orig_head        = model.classifier
        model.classifier = nn.Identity()
        emb              = model(tensor).cpu().numpy().astype("float32")
        model.classifier = orig_head
    faiss.normalize_L2(emb)

    # Search top-3 genera, gather candidates per species
    from collections import defaultdict
    species_pool = defaultdict(list)

    for g_prob, g_idx in zip(top3_p.tolist(), top3_i.tolist()):
        genus     = genera[g_idx]
        g_indices = genus_to_idx.get(genus, [])
        if not g_indices:
            continue
        g_vecs  = np.array([faiss_index.reconstruct(i) for i in g_indices], dtype="float32")
        sub_idx = faiss.IndexFlatIP(g_vecs.shape[1])
        sub_idx.add(g_vecs)
        k       = min(50, len(g_indices))   # fetch more to get 5 photos/species
        D, I    = sub_idx.search(emb, k)
        for score, local_i in zip(D[0], I[0]):
            rec = metadata[g_indices[local_i]]
            combined = g_prob * float(score)
            species_pool[rec["species"]].append({
                "genus":      genus,
                "genus_conf": g_prob,
                "score":      combined,
                "faiss_score":float(score),
                "path":       rec["path"],
            })

    # For each species: sort photos by combined score, keep top-5
    for sp in species_pool:
        species_pool[sp].sort(key=lambda x: x["score"], reverse=True)
        species_pool[sp] = species_pool[sp][:photos_per_species]

    # Rank species by their best photo score, take top-5
    ranked = sorted(species_pool.items(),
                    key=lambda kv: kv[1][0]["score"], reverse=True)[:top_n_species]

    results = []
    for sp, photos in ranked:
        results.append({
            "species":    sp,
            "genus":      photos[0]["genus"],
            "genus_conf": photos[0]["genus_conf"],
            "best_score": photos[0]["score"],
            "photos":     photos,
        })

    genus_labels = {genera[i.item()]: float(p.item()) for p, i in zip(top5_p, top5_i)}
    return results, genus_labels

def predict(img):
    if img is None:
        return {}, [], ""

    results, genus_labels = identify(img)

    # Build gallery: 5 species x up to 5 photos, grouped with caption
    gallery = []
    for rank, res in enumerate(results, 1):
        for j, photo in enumerate(res["photos"], 1):
            caption = (f"#{rank} {res['species']}  [{j}/{len(res['photos'])}]\n"
                       f"Genus: {res['genus']} ({res['genus_conf']*100:.0f}%)  |  "
                       f"Score: {photo['score']:.3f}")
            try:
                pil_img = Image.open(remap(photo["path"])).convert("RGB")
            except:
                pil_img = Image.new("RGB", (150,150), (220,220,220))
            gallery.append((pil_img, caption))

    best   = results[0] if results else None
    summary = (f"Best match: {best['species']}  (score {best['best_score']:.3f})"
               if best else "No results")

    return genus_labels, gallery, summary

with gr.Blocks(title="ONG Orchid Identifier") as demo:
    gr.Markdown(
        "# ONG Orchid Identifier\n"
        "Upload an orchid photo. Stage 1 predicts genus; "
        "Stage 2 returns the **5 most similar species** — each shown with up to **5 matching photos** "
        "ranked by visual similarity (top-3 genera searched)."
    )
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload Orchid Photo")
            run_btn   = gr.Button("Identify", variant="primary")
        with gr.Column(scale=2):
            genus_out   = gr.Label(num_top_classes=5, label="Stage 1 — Genus Prediction")
            summary_out = gr.Textbox(label="Best Match", interactive=False)

    gallery_out = gr.Gallery(
        label="Stage 2 — Top-5 Species x 5 Photos  (25 photos total, sorted by similarity)",
        columns=5, rows=5, height=600, object_fit="cover"
    )

    run_btn.click(fn=predict, inputs=img_input,
                  outputs=[genus_out, gallery_out, summary_out])
    img_input.change(fn=predict, inputs=img_input,
                     outputs=[genus_out, gallery_out, summary_out])

demo.launch(share=True, debug=False)
