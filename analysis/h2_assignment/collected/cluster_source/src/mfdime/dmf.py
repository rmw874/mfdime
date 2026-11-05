"""DMF's model classes and released checkpoints (github.com/kyungmnlee/dmf, huggingface.co/kyungmnlee/DMF).

The clone is not vendored (the repository has no licence file): it is passed as --dmf-repo and must be at DMF_SHA.
Its models/ package is imported as `dmf_models` (not as a top-level `models`) without the flash-attention kernels,
which need flash_attn CUDA builds; the torch_sdpa attention used here does not.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

DMF_SHA = "a6db0f6135e18f4a5dad6f38df3947ed61960742"  # main on 2026-10-29
NULL_CLASS = 1000  # the classifier-free-guidance label ("told nothing")
LATENT_SCALE = 0.18215  # generate.py: latents = vae latents * 0.18215, bias 0
DMF_DEPTH_XL = 20  # README: generate.py --dmf-depth=20 for DMF-XL/2


def import_models(repo: Path) -> types.ModuleType:
    if "dmf_models" in sys.modules:
        return sys.modules["dmf_models"]
    pkg_dir = repo / "models"
    if not (pkg_dir / "dmft.py").exists():
        raise SystemExit(f"STOP: {pkg_dir}/dmft.py not found; clone github.com/kyungmnlee/dmf at {DMF_SHA}")

    def flash_attn_func(*args, **kwargs):
        raise NotImplementedError("flash attention is stubbed out; build the models with attn_func='torch_sdpa'")

    for name in ("flash_attention_2_jvp", "flash_attention_3_jvp"):
        stub = types.ModuleType(f"dmf_models.{name}")
        stub.__dict__["flash_attn_func"] = flash_attn_func
        sys.modules[stub.__name__] = stub
    spec = importlib.machinery.ModuleSpec("dmf_models", None, is_package=True)
    spec.submodule_search_locations = [str(pkg_dir)]
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["dmf_models"] = pkg
    write_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # upstream commits its .pyc: leave them alone
    try:
        importlib.import_module("dmf_models.dmft")
        importlib.import_module("dmf_models.sit")
    finally:
        sys.dont_write_bytecode = write_bytecode
    return pkg


def build_dmf(models: types.ModuleType, name: str = "DMFT-XL/2", dmf_depth: int = DMF_DEPTH_XL) -> nn.Module:
    """use_logvar: the released checkpoint carries the training-time loss-weighting head (logvar_linear); it is unused
    unless forward(..., return_logvar=True), but building it lets the checkpoint load with an exact key match."""
    return models.dmft.DMFT_models[name](input_size=32, num_classes=1000, use_cfg_embedding=True, dmf_depth=dmf_depth,
                                         use_logvar=True, attn_func="torch_sdpa", qk_norm=False)


def build_sit(models: types.ModuleType, name: str = "SiT-XL/2") -> nn.Module:
    return models.sit.SiT_models[name](input_size=32, num_classes=1000, use_cfg_embedding=True, attn_func="torch_sdpa",
                                       qk_norm=False)


def load_weights(model: nn.Module, path: Path) -> dict[str, list[str]]:
    """Load a released checkpoint, refusing a partial load (generate.py's strict=False would silently load nothing
    from a wrapped state dict). Allowed: REPA's projector heads (unused at sampling) and the fixed sin-cos pos_embed."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("ema", "model", "state_dict"):
        if isinstance(sd.get(key), dict):
            sd = sd[key]
            break
    sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "pos_embed"]
    bad = [k for k in unexpected if not k.startswith("projectors.")]
    if missing or bad:
        raise SystemExit(f"STOP: {path}: {len(missing)} missing keys {missing[:4]}, {len(bad)} unexpected {bad[:4]}")
    return {"unexpected_ignored": sorted({k.split(".")[0] for k in unexpected})}


class Autoencoder(nn.Module):
    """SD-VAE ft-ema, the latent space SiT and DMF were trained in (generate.py: latents x LATENT_SCALE, bias 0)."""

    def __init__(self, vae: Any):  # diffusers' AutoencoderKL, whose lazy import types cannot follow
        super().__init__()
        self.vae = vae

    def encode(self, img: Tensor) -> Tensor:
        return self.vae.encode(img).latent_dist.mean * LATENT_SCALE

    def decode(self, z: Tensor) -> Tensor:
        return self.vae.decode(z / LATENT_SCALE).sample
