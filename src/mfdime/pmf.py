"""FiRe's pixel MeanFlow (pMF) for CelebA-HQ faces (Zeng et al., "FiRe: Fixed-Noise Refinement for Visual Counterfactual
Explanations", arXiv:2608.08664; pMF: Lu et al., arXiv:2601.22158). The repository is linked as the submodule
third_party/fire at FIRE_SHA and its src/pmf/dit.py is imported by path, never copied (it has no licence yet). The
weights are DapaoZeng/FiRe's smile checkpoint on Hugging Face at WEIGHTS_REV (CC BY-NC 4.0).

Conventions (FiRe's src/pmf/pixel_mean_flow.py): pixels in [-1, 1], z_t = (1 - t) x + t eps, v = eps - x, the same
path as DMF and SiT. The network predicts the clean image x_hat = net(z_t, t, r, y), conditioned on t - r only, and
the average velocity over [r, t] is u = (z_t - x_hat) / t, so z_r = z_t + (r - t) u as in DMF. At r = t that is the
instantaneous velocity, which FiRe trains on half of every batch: the same weights walk (Velocity) and jump (FlowMap),
so pMF is its own twin. Class tokens 0 and 1 follow FiRe's sorted class folders (non-smile/smile: 1 = smiling, as
STEEX's classifier reads it); NULL_CLASS is the dropout token. The auxiliary head, a pixel-space
smoother FiRe applies only to final outputs, is PixelSpace's decode, so every peek and every counterfactual gets it
once, as in FiRe's samplers. Velocity and FlowMap clip x_hat to [-1, 1], the clean estimate DiME's sampler clips
(guided-diffusion's p_mean_variance with clip_denoised=True), so every walk, jump and guided step on faces uses it.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

FIRE_SHA = "0840cdc0391d36c57371f3d1b2e8b4187663ea43"  # main on 2026-10-30
WEIGHTS_REV = "90de30efa18f0d84a45067d077880caa1139553b"  # DapaoZeng/FiRe on 2026-10-30
NULL_CLASS = 2
T_MIN = 1e-4  # FiRe clamps t here before dividing by it


def import_dit(repo: Path) -> types.ModuleType:
    if "fire_pmf_dit" in sys.modules:
        return sys.modules["fire_pmf_dit"]
    path = repo / "src" / "pmf" / "dit.py"
    if not path.exists():
        raise SystemExit(f"STOP: {path} not found; clone github.com/Yan-Zen9/FiRe at {FIRE_SHA}")
    spec = importlib.util.spec_from_file_location("fire_pmf_dit", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fire_pmf_dit"] = mod
    write_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # keep the submodule's tree clean
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = write_bytecode
    return mod


def build(mod: types.ModuleType, smoke: bool = False) -> nn.Module:
    """FiRe's CelebA-HQ 256 DiT-B/16 (configs/pMF-CelebAHQ256-smile.yaml); smoke: a tiny one of the same kind."""
    if smoke:
        return mod.DiT(input_size=256, patch_size=32, hidden_size=64, depth=2, num_heads=2, num_classes=3,
                       bottleneck_dim=16, aux_depth=2, aux_mid_channels=8)
    return mod.DiT(input_size=256, patch_size=16, in_channels=3, hidden_size=768, depth=12, num_heads=12, mlp_ratio=4.0,
                   class_dropout_prob=0.1, num_classes=3, learn_sigma=False, bottleneck_dim=128, aux_depth=8,
                   aux_mid_channels=128)


def load_weights(model: nn.Module, path: Path) -> dict:
    """FiRe's checkpoints are the DiT's state dict, possibly under 'model.' (CFgenerating/core.py strips it): exact key
    match."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if sum(k.startswith("model.") for k in sd) * 2 >= len(sd):
        sd = {k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"STOP: {path}: {len(missing)} missing keys {missing[:4]}, {len(unexpected)} unexpected {unexpected[:4]}")
    return {"keys": len(sd), "weights_rev": WEIGHTS_REV}


def _per_sample(t: Tensor) -> Tensor:
    return t.clamp_min(T_MIN).view(-1, 1, 1, 1)


class Velocity(nn.Module):
    """pMF at r = t in the SiT calling convention model(x, t, y): v = (x - x_hat) / t, x_hat clipped to [-1, 1]."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        return (x - self.net(x, t, t, y).clamp(-1, 1)) / _per_sample(t)


class FlowMap(nn.Module):
    """pMF in DMF's calling convention model(x, t, r, y): the average velocity over [r, t], u = (x - x_hat) / t, x_hat
    clipped to [-1, 1]."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: Tensor, t: Tensor, r: Tensor, y: Tensor) -> Tensor:
        return (x - self.net(x, t, r, y).clamp(-1, 1)) / _per_sample(t)


class PixelSpace(nn.Module):
    """The autoencoder interface for a pixel-space model: encode is the identity, decode is pMF's auxiliary head."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.aux = cast(nn.Module, net.aux_head)

    def encode(self, img: Tensor) -> Tensor:
        return img

    def decode(self, z: Tensor) -> Tensor:
        return self.aux(z)
