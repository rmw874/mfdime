"""Download H2's ImageNet assets on an internet-enabled cluster setup/login node.

Accept the ImageNet dataset terms and run `uv run hf auth login` first. Sources and paths come from mfdime.config.
This downloads the full ImageNet validation split and large model weights; rerunning reuses completed downloads.
Initialise third_party/dmf separately with `git submodule update --init third_party/dmf`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet50_Weights, VGG19_Weights, ViT_B_16_Weights

from mfdime import config


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    downloads = (
        (config.DMF_HF_REPO, config.DMF_REV, "model", config.DMF_CKPT.parent,
         [config.DMF_CKPT.name, config.SIT_CKPT.name]),
        (config.VAE_HF_REPO, config.VAE_REV, "model", config.IMAGENET_VAE,
         ["config.json", "diffusion_pytorch_model.*"]),
        (config.IMAGENET_HF_REPO, config.IMAGENET_REV, "dataset", config.IMAGENET_VAL.parent,
         ["data/validation-*"]),
    )
    for repo, rev, repo_type, folder, patterns in downloads:
        print(f"Download {repo}@{rev} -> {folder}", flush=True)
        snapshot_download(repo_id=repo, revision=rev, repo_type=repo_type, local_dir=folder, allow_patterns=patterns)
    for weights in (ResNet50_Weights.IMAGENET1K_V1, ViT_B_16_Weights.IMAGENET1K_V1, VGG19_Weights.IMAGENET1K_V1):
        print(f"Cache {weights}", flush=True)
        state = load_state_dict_from_url(weights.url, map_location="cpu", check_hash=True)
        del state
    required = [config.DMF_CKPT, config.SIT_CKPT, config.IMAGENET_VAE / "config.json"]
    missing = [str(p) for p in required if not p.is_file()]
    vae_weights = [*config.IMAGENET_VAE.glob("diffusion_pytorch_model.*")]
    shards = list(config.IMAGENET_VAL.glob("validation-*.parquet"))
    if missing or not vae_weights or len(shards) != 14:
        raise SystemExit(f"STOP: incomplete download: missing={missing}, VAE files={vae_weights}, shards={len(shards)}/14")
    print(f"Assets ready under {Path.cwd()}. Initialise the pinned DMF submodule before submitting jobs.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
