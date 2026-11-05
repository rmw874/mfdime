"""Shared run settings. Every script takes its defaults from here; a command-line flag overrides one for a single run.
DiME's own constants (scales, loss weights, noise levels) are the method, not settings, and stay in mfdime.dime."""

from __future__ import annotations

from pathlib import Path

SEEDS = (0, 1, 2, 3, 4)
N_PRACTICE = 16  # the first 16 photos of each dataset: the gain is chosen on them, the held-out photos come after
N_PHOTOS = 16  # photos per run (--n)
BATCH = 8
GAIN = {"imagenet": 1.0, "celebahq": 3.0}
PEEK = "jump1"
H2_PAIRS = ((340, 339), (293, 286), (285, 283))
H2_ARMS = (("jump1", "shortcut"), ("jump1", "exact"), ("jump4", "shortcut"), ("jump4", "exact"))
H2_SEED_GROUPS = ((0, 1), (2, 3), (4,))

IMAGENET_SOURCE, IMAGENET_TARGET = 340, 339  # zebra -> sorrel, the first of ACE's ImageNet pairs
ATTRIBUTE = "smile"  # the CelebA-HQ attribute: STEEX's classifier output and the pMF checkpoint's classes

# H2's login-node downloader uses these pinned sources, with the paths below.
DMF_HF_REPO, DMF_REV = "kyungmnlee/DMF", "4b040eb1ed1e3f9e1d01dd89d499f720296fba98"
VAE_HF_REPO, VAE_REV = "stabilityai/sd-vae-ft-ema", "f04b2c4b98319346dad8c65879f680b1997b204a"
IMAGENET_HF_REPO, IMAGENET_REV = "ILSVRC/imagenet-1k", "49e2ee26f3810fb5a7536bbf732a7b07389a47b5"

# Where the downloads live, and where each comes from. Hugging Face items download with
#   uv run hf download <repo> [files] --revision <rev> --local-dir <dir>
# adding --repo-type dataset for datasets (ImageNet is gated: accept its terms and run uv run hf auth login first).
# torchvision (ResNet-50, ViT-B/16, VGG19: IMAGENET1K_V1) and pytorch_fid (Inception) fetch their weights into the torch
# hub cache on first use, which needs a machine with internet access.
DMF_REPO = Path("third_party/dmf")  # git submodule
# kyungmnlee/DMF at 4b040eb1ed1e3f9e1d01dd89d499f720296fba98 (MIT, model card)
DMF_CKPT = Path("weights/dmf/dmf_xl_2_256.pt")
SIT_CKPT = Path("weights/dmf/sit_xl_2_repa_256.pt")
# stabilityai/sd-vae-ft-ema at f04b2c4b98319346dad8c65879f680b1997b204a (MIT)
IMAGENET_VAE = Path("weights/sd-vae-ft-ema")
# ILSVRC/imagenet-1k (dataset) at 49e2ee26f3810fb5a7536bbf732a7b07389a47b5, --include "data/validation-*" (ImageNet terms)
IMAGENET_VAL = Path("data/imagenet-1k/data")
FIRE_REPO = Path("third_party/fire")  # git submodule
# DapaoZeng/FiRe at 90de30efa18f0d84a45067d077880caa1139553b (CC BY-NC 4.0)
PMF_CKPT = Path("weights/fire/pmf-celebahq256-smile.pt")
# liusq/CelebAMask-HQ (dataset) at 9cd422d28a64c1f5e7144f3a5a398b78bbb9ad11 (CelebAMask-HQ terms, non-commercial research)
CELEBAHQ_ZIP = Path("data/celebahq/CelebAMask-HQ.zip")
# noname110/celeba (dataset) at 438b96fa79375870b726cd408a3da1eda4f074c4 (CelebA terms, non-commercial research)
CELEBA_PARTITION = Path("data/celebahq/list_eval_partition.txt")
# github.com/valeoai/STEEX release v.0.1, checkpoints_decision_densenet.tar.gz, unpacked (STEEX repo terms)
STEEX_CKPT = Path("weights/steex/decision_densenet/celebamaskhq/checkpoint.tar")
