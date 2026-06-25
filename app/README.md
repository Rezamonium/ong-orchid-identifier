---
title: ONG Orchid Identifier v3
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.15.2
python_version: '3.13'
app_file: app.py
pinned: false
license: mit
short_description: DINOv2 genus + FAISS species ID for Bird's Head orchids
---

# New Guinea Orchid Identifier v3

Two-stage identifier for orchids of the Bird's Head Peninsula (West Papua):

1. **DINOv2 ViT-L/14** genus classifier (120 genera, @448px)
2. **FAISS** visual-similarity retrieval for the most similar species

Set the `HF_TOKEN` secret in Space settings to enable consented community photo
uploads to `Rezamonium/birdshead-community-photos`.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
