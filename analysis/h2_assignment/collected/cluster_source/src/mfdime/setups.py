"""Everything a DiME run needs, built once per dataset: the generator twins (a walker's velocity and a flow map), the
autoencoder, the guided classifier, an independent check classifier where one exists, DiME's perceptual loss, and the
photos with their targets.

  imagenet  SiT-XL/2+REPA (walker) and DMF-XL/2 (flow map), released as twins; SD-VAE ft-ema; ResNet-50 (guided),
            ViT-B/16 (check); zebra -> sorrel by default; DiME's multi-class loss (log-softmax)
  celebahq  FiRe's pixel MeanFlow for smiles, a DiT-B/16 that walks (r = t) and jumps with the same weights, in pixels;
            STEEX's DenseNet on smiling (guided), no check classifier; target = the opposite of the classifier's answer
            (DiME/ACE); DiME's CelebA loss form (logits)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from mfdime import celebahq, config, dime, dmf, flow, imagenet, pmf
from mfdime.common import predict

DATASETS = ("imagenet", "celebahq")


def _full(x: Tensor, value: float) -> Tensor:
    return torch.full((x.shape[0],), value, device=x.device, dtype=x.dtype)


def _labels(x: Tensor, label: int) -> Tensor:
    return torch.full((x.shape[0],), label, device=x.device, dtype=torch.long)


@dataclass
class Twins:
    walker: nn.Module  # walker(x, t, y): the velocity
    flow_map: nn.Module  # flow_map(x, t, r, y): the average velocity over [r, t]

    def v(self, label: int) -> flow.Velocity:
        return lambda x, t: self.walker(x, _full(x, t), _labels(x, label))

    def u(self, label: int) -> flow.MeanVelocity:
        return lambda x, t, r: self.flow_map(x, _full(x, t), _full(x, r), _labels(x, label))


@dataclass
class Setup:
    dataset: str
    twins: Twins
    ae: dmf.Autoencoder | pmf.PixelSpace  # .encode(img) and .decode(z): latents (imagenet) or pixels (celebahq)
    guide: nn.Module  # [-1, 1] images -> logits (binary attributes as two columns [0, l])
    check: nn.Module | None  # an independent classifier that never guides the edits
    perc: dime.Perceptual
    use_logits: bool
    null_label: int  # the generator's label for "told nothing" (classifier-free guidance's null class)
    info: dict


@dataclass
class Photos:
    names: list[str]
    x: Tensor  # (N, 3, 256, 256) in [-1, 1]
    target: Tensor  # (N,) the class each counterfactual aims at


def add_args(p: argparse.ArgumentParser) -> None:
    """The model, data and task arguments both datasets understand (defaults: mfdime.config)."""
    p.add_argument("--dataset", choices=DATASETS, default="imagenet")
    p.add_argument("--dmf-repo", type=Path, default=config.DMF_REPO)
    p.add_argument("--dmf-ckpt", type=Path, default=config.DMF_CKPT)
    p.add_argument("--sit-ckpt", type=Path, default=config.SIT_CKPT)
    p.add_argument("--vae", type=Path, default=config.IMAGENET_VAE)
    p.add_argument("--imagenet-val", type=Path, default=config.IMAGENET_VAL)
    p.add_argument("--source", type=int, default=config.IMAGENET_SOURCE)
    p.add_argument("--target", type=int, default=config.IMAGENET_TARGET)
    p.add_argument("--fire-repo", type=Path, default=config.FIRE_REPO)
    p.add_argument("--pmf-ckpt", type=Path, default=config.PMF_CKPT)
    p.add_argument("--celebahq-zip", type=Path, default=config.CELEBAHQ_ZIP)
    p.add_argument("--celeba-partition", type=Path, default=config.CELEBA_PARTITION)
    p.add_argument("--steex-ckpt", type=Path, default=config.STEEX_CKPT)


def required_files(args) -> list[Path]:
    if args.dataset == "imagenet":
        return [args.dmf_repo / "models/dmft.py", args.dmf_ckpt, args.sit_ckpt, args.vae]
    return [args.fire_repo / "src/pmf/dit.py", args.pmf_ckpt, args.celebahq_zip, args.celeba_partition, args.steex_ckpt]


def load_vgg19(smoke: bool, device: str) -> dime.Perceptual:
    from torchvision.models import VGG19_Weights, vgg19

    try:  # from the torch hub cache
        net = vgg19(weights=None if smoke else VGG19_Weights.IMAGENET1K_V1)
    except OSError as e:
        raise SystemExit(f"STOP: VGG19 weights not in the torch hub cache ({e})") from e
    return dime.Perceptual(net).eval().requires_grad_(False).to(device)


def _freeze(*mods: nn.Module, device: str) -> None:
    for m in mods:
        m.eval().requires_grad_(False).to(device)


def build(args) -> Setup:
    """args: add_args' fields plus smoke (bool) and device."""
    missing = [] if args.smoke else [str(f) for f in required_files(args) if not f.exists()]
    if missing:
        raise SystemExit(f"STOP: missing {', '.join(missing)} (download list: src/mfdime/config.py)")
    perc = load_vgg19(args.smoke, args.device)
    return _imagenet(args, perc) if args.dataset == "imagenet" else _celebahq(args, perc)


def _imagenet(args, perc: dime.Perceptual) -> Setup:
    from diffusers import AutoencoderKL
    from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16

    models = dmf.import_models(args.dmf_repo)
    if args.smoke:
        walker, flow_map = dmf.build_sit(models, "SiT-S/8"), dmf.build_dmf(models, "DMFT-S/8", dmf_depth=8)
        vae = AutoencoderKL(block_out_channels=(8, 8, 8, 8), down_block_types=("DownEncoderBlock2D",) * 4,
                            up_block_types=("UpDecoderBlock2D",) * 4, norm_num_groups=4)
        resnet, vit, info = resnet50(), vit_b_16(), {"smoke": "random weights"}
    else:
        walker, flow_map = dmf.build_sit(models), dmf.build_dmf(models)
        info = {"sit": dmf.load_weights(walker, args.sit_ckpt), "dmf": dmf.load_weights(flow_map, args.dmf_ckpt)}
        vae = AutoencoderKL.from_pretrained(args.vae)
        try:  # from the torch hub cache
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        except OSError as e:
            raise SystemExit(f"STOP: classifier weights not in the torch hub cache ({e})") from e
    ae, guide, check = dmf.Autoencoder(vae), imagenet.Classifier(resnet), imagenet.Classifier(vit)
    _freeze(walker, flow_map, ae, guide, check, device=args.device)
    return Setup("imagenet", Twins(walker, flow_map), ae, guide, check, perc, use_logits=False,
                 null_label=dmf.NULL_CLASS, info=info)


def _celebahq(args, perc: dime.Perceptual) -> Setup:
    clf = celebahq.DecisionDensenet() if args.smoke else celebahq.load_classifier(args.steex_ckpt)
    guide = celebahq.BinaryAttribute(clf, config.ATTRIBUTE)
    net = pmf.build(pmf.import_dit(args.fire_repo), smoke=args.smoke)
    info = {"smoke": "random weights"} if args.smoke else {"pmf": pmf.load_weights(net, args.pmf_ckpt)}
    ae = pmf.PixelSpace(net)
    _freeze(net, ae, guide, device=args.device)
    return Setup("celebahq", Twins(pmf.Velocity(net), pmf.FlowMap(net)), ae, guide, None, perc, use_logits=True,
                 null_label=pmf.NULL_CLASS, info=info)


def _split(pool: list, split: str, n: int, skip: int = 0) -> list:
    """n photos of a split in the pool's order, after its first `skip` (n <= 0: all of them): "practice" is the
    pool's first config.N_PRACTICE photos (where the gain is chosen), "held_out" the photos after them (for the runs
    that decide), so the two never overlap and stay put whatever n is."""
    start, stop = (0, config.N_PRACTICE) if split == "practice" else (config.N_PRACTICE, len(pool))
    chosen = pool[start:stop][skip:]
    if n > 0 and len(chosen) < n:
        raise SystemExit(f"STOP: {split} has {len(chosen)} photos after skipping {skip}, --n asks for {n} "
                         f"(practice = the first {config.N_PRACTICE}, held_out = the rest)")
    return chosen[:n] if n > 0 else chosen


@torch.no_grad()
def photos(args, s: Setup, split: str, n: int, skip: int = 0) -> tuple[Photos, dict]:
    """The photos to explain and a record of how they were chosen. Each dataset has one pool in a fixed order, cut
    into practice and held-out photos by _split (skip: leave out the split's first photos; n <= 0: all the rest).

    imagenet: the --source validation photos the guide classifies correctly; target = --target.
    celebahq: CelebA's test partition by CelebA-HQ index; target = the opposite of the guide's answer on the attribute
              (DiME/ACE).
    """
    if s.dataset == "imagenet":
        data = imagenet.load_classes(args.imagenet_val, [args.source, args.target])
        items = data[args.source]
        imgs = torch.stack([t for _, t in items])
        pred = predict(s.guide, imgs, config.BATCH, args.device)
        ok = [i for i in range(len(items)) if args.smoke or pred[i] == args.source]
        chosen = _split(ok, split, n, skip)
        names = [items[i][0] for i in chosen]
        return Photos(names, imgs[chosen], torch.full((len(chosen),), args.target)), \
            {"split": split, "skip": skip, "candidates": len(items), "correct": len(ok), "names": names}
    faces = celebahq.load_split(args.celebahq_zip, args.celeba_partition, "test")
    chosen = _split(faces, split, n, skip)
    x = celebahq.load_images(args.celebahq_zip, [f.idx for f in chosen])
    pred = predict(s.guide, x, config.BATCH, args.device)
    names = [f"{f.idx}.jpg" for f in chosen]
    return Photos(names, x, 1 - pred), \
        {"split": split, "skip": skip, "test_faces": len(faces), "attribute": config.ATTRIBUTE, "names": names,
         "label": [f.attrs[config.ATTRIBUTE] for f in chosen], "guide_pred": pred.tolist(),
         "guide_accuracy": float((pred == torch.tensor([f.attrs[config.ATTRIBUTE] for f in chosen])).float().mean())}
