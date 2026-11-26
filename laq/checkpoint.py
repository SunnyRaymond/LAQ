"""The LAQ checkpoint: packed 4-bit weights and the calibrated allocation.

A checkpoint is a directory with
  config.json            model identity, token layout, rotation seed, the allocation (refined tokens per class) and the
                         offline gaze
  weights.safetensors    for every quantized Linear `<name>`: `<name>.packed` (uint8, two 4-bit codes per byte) and
                         `<name>.scales` (FP32, one per output row)
The layers that stay at 16 bits (patch embedding, heads, time conditioning), the VAE and the text encoder come from the
original model.
"""
from __future__ import annotations

import json
from pathlib import Path

FORMAT = "laq-v1"


def save(path, config, weights):
    """weights: {module name: bank}, a bank as returned by laq.rotations.fit."""
    from safetensors.torch import save_file
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tensors, layers = {}, {}
    for name, bank in weights.items():
        if bank["kind"] != "sym" or bank["bits"] != 4 or bank["group"] != bank["shape"][1]:
            raise ValueError(f"{name}: expected per-row symmetric 4-bit codes")
        tensors[f"{name}.packed"] = bank["packed"].cpu().contiguous()
        tensors[f"{name}.scales"] = bank["scales"].float().cpu().contiguous()
        layers[name] = [int(v) for v in bank["shape"]]
    save_file(tensors, str(path / "weights.safetensors"))
    (path / "config.json").write_text(json.dumps(dict(config, format=FORMAT, layers=layers), indent=2) + chr(10),
                                      encoding="utf-8")


def load(path, device="cpu"):
    """Returns (config, {module name: bank})."""
    from safetensors.torch import load_file
    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("format") != FORMAT:
        raise ValueError(f"Unsupported checkpoint format {config.get('format')!r}")
    tensors = load_file(str(path / "weights.safetensors"), device=str(device))
    weights = {}
    for name, shape in config["layers"].items():
        weights[name] = dict(kind="sym", bits=4, group=shape[1], shape=shape, packed=tensors[f"{name}.packed"],
                             scales=tensors[f"{name}.scales"])
    return config, weights


def resolve(checkpoint, device="cpu"):
    """A (config, weights) pair, a local directory, or the name of a released model."""
    if isinstance(checkpoint, tuple):
        return checkpoint
    if Path(checkpoint).is_dir():
        return load(checkpoint, device)
    from .hub import download
    return load(download(str(checkpoint)), device)
