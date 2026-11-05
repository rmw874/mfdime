"""CelebA-HQ faces as ACE evaluates them (Jeanneret et al., CVPR 2023; github.com/guillaumejs2403/ACE, MIT):
CelebAMask-HQ (Lee et al., CVPR 2020) read straight from its zip, CelebA's test partition through the HQ-to-CelebA
mapping (guided_diffusion/image_datasets.py: CelebAHQDataset), images resized to 256 px, and STEEX's decision
classifier (Jacob et al., ECCV 2022; github.com/valeoai/STEEX release v.0.1, checkpoints_decision_densenet:
decision_densenet/celebamaskhq/checkpoint.tar), a DenseNet-121 with three outputs (male, smile, young; ACE models/__init__.py).
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

ATTRIBUTES = ("male", "smile", "young")  # STEEX's three outputs, in order (CelebA attributes 20, 31, 39)
CELEBA_NAMES = {"male": "Male", "smile": "Smiling", "young": "Young"}
ROOT = "CelebAMask-HQ/"


@dataclass
class Face:
    idx: int  # CelebA-HQ index (CelebA-HQ-img/<idx>.jpg)
    attrs: dict[str, int]  # 1 / 0 for the three attributes


def load_split(zip_path: Path, partition_path: Path, split: str = "test") -> list[Face]:
    """The faces of one CelebA partition (train 0, val 1, test 2), sorted by CelebA-HQ index."""
    part = {"train": 0, "val": 1, "test": 2}[split]
    partition = {}
    for line in partition_path.read_text().split("\n"):
        if line.strip():
            name, p = line.split()[:2]
            partition[name] = int(p)
    with zipfile.ZipFile(zip_path) as z:
        anno = z.read(ROOT + "CelebAMask-HQ-attribute-anno.txt").decode().split("\n")
        mapping = z.read(ROOT + "CelebA-HQ-to-CelebA-mapping.txt").decode().split("\n")
    names = anno[1].split()
    cols = {a: names.index(CELEBA_NAMES[a]) for a in ATTRIBUTES}
    attrs = {}
    for row in anno[2:]:
        if row.strip():
            parts = row.split()
            vals = parts[1:]
            attrs[int(parts[0].removesuffix(".jpg"))] = {a: int(vals[c] == "1") for a, c in cols.items()}
    faces = []
    for row in mapping[1:]:
        if row.strip():
            idx, _, orig_file = row.split()[:3]
            if partition.get(orig_file) == part:
                faces.append(Face(int(idx), attrs[int(idx)]))
    return sorted(faces, key=lambda f: f.idx)


def load_images(zip_path: Path, idxs: list[int], size: int = 256) -> Tensor:
    """(N, 3, size, size) in [-1, 1]: ACE's Resize(size) + CenterCrop(size) (the HQ images are square)."""
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for i in idxs:
            img = Image.open(io.BytesIO(z.read(f"{ROOT}CelebA-HQ-img/{i}.jpg"))).convert("RGB")
            img = img.resize((size, size), resample=Image.Resampling.BILINEAR)
            out.append(torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1)
    return torch.stack(out)


def fake_dataset(root: Path, n: int = 6, seed: int = 0) -> tuple[Path, Path]:
    """A CelebAMask-HQ zip and a CelebA partition file in the real layout, with noise faces, every one in the test
    partition (for the smoke runs). Returns (zip path, partition path)."""
    rng = np.random.default_rng(seed)
    names = [f"Attr{i}" for i in range(40)]
    for a, col in (("Male", 20), ("Smiling", 31), ("Young", 39)):
        names[col] = a
    root.mkdir(parents=True, exist_ok=True)
    zip_path, part_path = root / "CelebAMask-HQ.zip", root / "list_eval_partition.txt"
    with zipfile.ZipFile(zip_path, "w") as z:
        rows, maps = [str(n), " ".join(names)], ["idx        orig_idx   orig_file"]
        for i in range(n):
            buf = io.BytesIO()
            Image.fromarray(rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)).save(buf, format="JPEG")
            z.writestr(f"{ROOT}CelebA-HQ-img/{i}.jpg", buf.getvalue())
            rows.append(f"{i}.jpg  " + " ".join(rng.choice(["1", "-1"], 40)))
            maps.append(f"{i}          {1000 + i}     {1001 + i:06d}.jpg")
        z.writestr(ROOT + "CelebAMask-HQ-attribute-anno.txt", "\n".join(rows) + "\n")
        z.writestr(ROOT + "CelebA-HQ-to-CelebA-mapping.txt", "\n".join(maps) + "\n")
    part_path.write_text("".join(f"{1001 + i:06d}.jpg 2\n" for i in range(n)) + "000001.jpg 0\n")
    return zip_path, part_path


class _DenseNet121(nn.Module):
    """STEEX's feature extractor (ACE models/steex/DecisionDensenetModel.py); the same module names, so its
    checkpoint loads as is."""

    def __init__(self):
        super().__init__()
        from torchvision.models import densenet121

        self.feat_extract = densenet121(weights=None)
        self.feat_extract.classifier = nn.Identity()
        self.output_size = 1024

    def forward(self, x: Tensor) -> Tensor:
        return self.feat_extract(x)


class DecisionDensenet(nn.Module):
    """STEEX's DecisionDensenetModel(num_classes=3): three attribute logits."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.feat_extract = _DenseNet121()
        self.classifier = nn.Linear(self.feat_extract.output_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.feat_extract(x))


def load_classifier(path: Path) -> DecisionDensenet:
    """STEEX's checkpoint dictionary holds the weights under 'model_state_dict' (ACE models/__init__.py)."""
    net = DecisionDensenet()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = net.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        raise SystemExit(f"STOP: {path}: {len(missing)} missing keys {missing[:4]}, {len(unexpected)} unexpected {unexpected[:4]}")
    return net


class BinaryAttribute(nn.Module):
    """One attribute of the three-output classifier as two-class logits [0, l] (class 1 = has the attribute), on
    [-1, 1] images: ACE feeds [0, 1] images through Normalizer(0.5, 0.5), which is the [-1, 1] image itself."""

    def __init__(self, net: nn.Module, attribute: str):
        super().__init__()
        self.net, self.i = net, ATTRIBUTES.index(attribute)

    def forward(self, img: Tensor) -> Tensor:
        logit = self.net(img)[:, self.i]
        return torch.stack([torch.zeros_like(logit), logit], 1)
