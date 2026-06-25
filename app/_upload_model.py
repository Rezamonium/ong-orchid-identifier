"""Upload the fp16 model + vocab to the public model repo. Resumable & idempotent.

Run tomorrow on good wifi:
    export HF_TOKEN=hf_yourNEWtoken      # (Git Bash)   or  $env:HF_TOKEN="..." (PowerShell)
    python _upload_model.py
Re-run safely if it drops — it skips files already fully uploaded.
"""
import os, sys
from huggingface_hub import HfApi

TOK = os.environ.get("HF_TOKEN")
if not TOK:
    sys.exit("Set HF_TOKEN first:  export HF_TOKEN=hf_...   (or $env:HF_TOKEN in PowerShell)")

REPO = "Rezamonium/ong-dinov2l-v3"
api  = HfApi(token=TOK)
api.create_repo(repo_id=REPO, repo_type="model", private=False, exist_ok=True)

jobs = [("models/dinov2l/best_model_fp16.pth", "best_model.pth", 600_000_000),
        ("models/dinov2l/vocab.json",          "vocab.json",     0)]
for local, remote, min_sz in jobs:
    try:
        info = api.get_paths_info(repo_id=REPO, repo_type="model", paths=[remote])
        sz = 0
        for m in info:
            sz = getattr(getattr(m, "lfs", None), "size", None) or getattr(m, "size", 0)
        if sz and sz >= max(min_sz, 1):
            print(f"skip {remote} (already there, {sz} bytes)"); continue
    except Exception:
        pass
    print(f"uploading {remote} ...", flush=True)
    api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                    repo_id=REPO, repo_type="model")
    print(f"  {remote} OK", flush=True)

print("files:", api.list_repo_files(repo_id=REPO, repo_type="model"))
print("DONE")
