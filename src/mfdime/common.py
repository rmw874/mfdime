"""Small helpers every script uses: batching, predictions, image files and the run's environment record."""

from __future__ import annotations

import subprocess
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.version
from PIL import Image
from torch import Tensor, nn


def batches(n: int, size: int) -> list[slice]:
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def key_seed(key: str) -> int:
    """A generator seed fixed by a string (for example a photo and a seed), the same on every machine and run."""
    return zlib.crc32(key.encode())


@torch.no_grad()
def predict(clf: nn.Module, imgs: Tensor, batch: int, device: str) -> Tensor:
    return torch.cat([clf(imgs[s].to(device)).argmax(1).cpu() for s in batches(len(imgs), batch)])


def save_png(img: Tensor, path: Path) -> None:
    arr = ((img.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)


def save_grid(rows: list[Tensor], path: Path, thumb: int = 128) -> None:
    """Rows of [-1, 1] images (each row a (k, 3, H, W) tensor) as one JPEG, thumbnails of `thumb` px."""
    from torchvision.utils import make_grid

    k = rows[0].shape[0]
    imgs = torch.cat([F.interpolate(r.float().cpu(), size=thumb, mode="bilinear", antialias=True) for r in rows])
    grid = make_grid(((imgs + 1) / 2).clamp(0, 1), nrow=k, padding=2, pad_value=1.0)
    Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(path, quality=90)


def repo_sha(repo: Path) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    except FileNotFoundError:  # no git on the node: recorded as unknown
        return None
    return r.stdout.strip() or None


def env_info(args, extra: dict) -> dict:
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "dmf_sha": repo_sha(args.dmf_repo), "fire_sha": repo_sha(args.fire_repo),
            "args": {k: str(v) for k, v in vars(args).items()}, **extra}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True)
        info["nvidia_smi"] = r.stdout.strip()
    except FileNotFoundError:
        info["nvidia_smi"] = None
    return info
