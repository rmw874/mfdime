"""Frozen, evaluation-only OpenCLIP smile scorer. Class 0 = neutral/non-smiling, class 1 = smiling.

The two scores are cosine similarities, not calibrated probabilities. All model files are loaded locally.
Keep this pilot's configuration separate from ImageNet's generation code/protocol so existing runs can resume.
"""

from __future__ import annotations

import json
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn

MODEL = "ViT-L-14"
REPO = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
REVISION = "1627032197142fbe2a7cfec626f4ced3ae60d07a"
DIRECTORY = Path("weights/openclip/ViT-L-14-laion2b")
WEIGHT_FILE = "open_clip_pytorch_model.safetensors"
PROMPTS = ("a photo of a person with a neutral facial expression.", "a photo of a person smiling.")


def to_pil(image: Tensor) -> Image.Image:
    """Same deterministic RGB quantisation for originals and generated images before OpenCLIP's own transform."""
    pixels = ((image.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(pixels)


class SmileEvaluator(nn.Module):
    text_features: Tensor

    def __init__(self, model: Any, preprocess, tokens: Tensor, device: str):
        super().__init__()
        self.model: Any = model.eval().requires_grad_(False).to(device)
        self.preprocess = preprocess
        with torch.no_grad():
            text = F.normalize(self.model.encode_text(tokens.to(device)).float(), dim=-1)
        self.register_buffer("text_features", text)
        self.eval()

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        x = torch.stack([self.preprocess(to_pil(img)) for img in images]).to(self.text_features.device)
        features = F.normalize(self.model.encode_image(x).float(), dim=-1)
        scores = features @ self.text_features.T
        if not torch.isfinite(scores).all():
            raise FloatingPointError("nonfinite OpenCLIP cosine scores")
        return scores


def load(directory: Path, device: str) -> tuple[SmileEvaluator, dict]:
    open_clip = import_module("open_clip")

    required = [directory / WEIGHT_FILE, directory / "open_clip_config.json", directory / "download.json"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SystemExit(f"STOP: missing {missing}; run python -m mfdime.download_clip_smile on the setup node")
    metadata = json.loads((directory / "download.json").read_text())
    if metadata.get("repo") != REPO or metadata.get("revision") != REVISION:
        raise ValueError("OpenCLIP download does not match the pinned checkpoint")
    cfg = json.loads((directory / "open_clip_config.json").read_text())
    if cfg["model_cfg"] != open_clip.get_model_config(MODEL):
        raise ValueError("OpenCLIP architecture differs from the downloaded checkpoint configuration")
    preprocessing = cfg["preprocess_cfg"]
    model, _, transform = open_clip.create_model_and_transforms(
        MODEL, pretrained=str(directory / WEIGHT_FILE), precision="fp32", device=device,
        image_mean=tuple(preprocessing["mean"]), image_std=tuple(preprocessing["std"]),
        image_interpolation="bicubic", image_resize_mode="shortest")
    tokens = open_clip.get_tokenizer(MODEL)(list(PROMPTS))
    evaluator = SmileEvaluator(model, transform, tokens, device)
    info = {"model": MODEL, "repo": REPO, "revision": REVISION, "open_clip_torch": version("open_clip_torch"),
            "prompts": list(PROMPTS), "class_order": ["neutral/non-smiling", "smiling"], "decision": "argmax cosine",
            "preprocessing": str(transform), "scores_are_probabilities": False, "input_quantisation": "RGB uint8"}
    return evaluator, info


def classification_report(labels: list[int], predictions: list[int]) -> dict:
    labels_array, pred_array = np.asarray(labels), np.asarray(predictions)
    if len(labels) != len(predictions) or not labels or not set(labels) <= {0, 1} or not set(predictions) <= {0, 1}:
        raise ValueError("expected nonempty paired binary labels and predictions")
    recalls = []
    classes = {}
    for cls in (0, 1):
        mask = labels_array == cls
        recall = float((pred_array[mask] == cls).mean()) if mask.any() else None
        classes[str(cls)] = {"count": int(mask.sum()), "recall": recall, "predicted": int((pred_array == cls).sum())}
        if recall is not None:
            recalls.append(recall)
    return {"n": len(labels), "accuracy": float((labels_array == pred_array).mean()), "classes": classes,
            "balanced_accuracy": sum(recalls) / 2 if len(recalls) == 2 else None,
            "confusion_true_rows_predicted_columns": [[int(((labels_array == a) & (pred_array == b)).sum())
                                                       for b in (0, 1)] for a in (0, 1)]}
