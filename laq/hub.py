"""The released LAQ checkpoints."""
from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path


def list_models():
    return json.loads(files("laq").joinpath("models.json").read_text(encoding="utf-8"))


def download(model, cache_dir=None, local_files_only=False):
    """Fetch one released checkpoint at its pinned revision and return its directory. The original model (16-bit
    interface layers, VAE, text encoder) is obtained separately and keeps its own license."""
    catalog = list_models()
    if model not in catalog:
        raise ValueError(f"Unknown model {model!r}; choose from {', '.join(catalog)}")
    spec = catalog[model]
    if not spec["repo_id"]:
        raise RuntimeError(f"The checkpoint of {model} is not released yet")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(spec["repo_id"], revision=spec["revision"], cache_dir=cache_dir,
                                  local_files_only=local_files_only,
                                  allow_patterns=["config.json", "weights.safetensors"]))
