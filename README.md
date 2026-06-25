# AnggrekScan

**Vision-Transformer genus identification and species retrieval for the Orchidaceae of New Guinea.**

This repository contains the code accompanying the manuscript:

> Saputra, et al. (2026)
> *Vision Transformers versus convolutional networks for fine-grained orchid genus
> identification in a species-rich, data-poor flora: a controlled benchmark on the
> Orchidaceae of New Guinea.* (Manuscript)

| | |
|---|---|
| 🔬 Live application | https://huggingface.co/spaces/Rezamonium/ong-orchid-identifierv3 |
| 🌐 Project website + multi-access key | https://birdsheadorchid.id |
| 📦 Archived release (code, weights, results) | Zenodo DOI: _to be assigned_ |
| 📄 Paper | DOI: _to be assigned_ |

## Overview

A two-stage system for identifying New Guinea orchids from field photographs:

1. **Genus classification** — a fine-tuned backbone predicts the genus of a query image.
2. **Species retrieval** — FAISS nearest-neighbour search over image embeddings returns
   visually similar reference species for expert verification.

The study is a controlled benchmark of four pretrained backbones under an identical
protocol — two Vision Transformers (**DINOv2**, **BioCLIP 2**) and two convolutional
networks (**ConvNeXt V2-L**, **EfficientNetV2-L**) — on a single, frozen,
species-stratified partition of field photographs spanning 120 genera and 1,350 species.
DINOv2 with embedding retrieval was the strongest and is the deployed backbone.

## Repository structure

```
pipeline/         End-to-end pipeline (Python): dataset audit, merging, split generation,
                  image-type detection, evaluation, figures, and the open-set protocol.
colab_scripts/    Training, FAISS index building, and app-build scripts run on Colab GPUs.
colab_notebooks/  Run notebooks: backbone bake-off, per-model pilots, temperature scaling,
                  and the open-set leave-K-genera-out experiment.
app/              Deployed Hugging Face Space application (Gradio).
```

> Model weights, derived results (per-genus accuracy, confusion matrices, figures), and
> data manifests are distributed with the **Zenodo** archive, not in this Git repository.

## Installation

```bash
git clone https://github.com/rezamonium/ong-orchid-identifier.git
cd ong-orchid-identifier
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r app/requirements.txt
```

Python 3.10+ is recommended. Key dependencies: `torch`, `timm`, `open_clip_torch`
(BioCLIP 2), `faiss-cpu`, `gradio`, `numpy`, `pandas`, `scikit-learn`.

## Usage

```bash
# Train a backbone (example) — see colab_scripts/ for the full bake-off
python colab_scripts/03_train_bakeoff_colab.py

# Build the FAISS species index from reference embeddings
python colab_scripts/04_build_faiss_colab.py

# Evaluate and reproduce per-genus metrics / confusion matrices
python pipeline/13_evaluate.py
python pipeline/17_compare_models.py

# Run the interactive application locally
python app/app.py
```

The training notebooks (`colab_notebooks/`) reproduce every reported result, including
temperature scaling and the open-set retraining experiment.

## Model weights

The deployed **DINOv2** backbone (~1.2 GB), the FAISS species index, and the reference
embeddings are not stored in Git. Obtain them from:

- the **Hugging Face Space** (served live), or
- the **Zenodo** archive (`model_weights/`, permanent DOI).

## Data availability

The raw orchid photographs are **not** redistributed. Reference imagery originates from
[orchidsnewguinea.com](https://www.orchidsnewguinea.com), [iNaturalist](https://www.inaturalist.org),
and contributed field collections, and is subject to third-party licensing that does not
permit bulk redistribution. The split manifests (file names + labels) are provided in the
Zenodo archive to support reproducibility.

## Citation

If you use this software, please cite it via [`CITATION.cff`](CITATION.cff) (GitHub's
“Cite this repository” button) **and** cite the associated paper once published.

## License

Code is released under the **MIT License** (add a `LICENSE` file). Note that model weights
and any derived data are subject to the licensing of the underlying images; see
*Data availability* above.

## Acknowledgements

We thank the contributors and curators of orchidsnewguinea.com, the iNaturalist community,
and the field photographers whose images made this work possible; and Jeffrey Champion for contributing
photographs. This work was supported by the Australian Orchid Foundation.
