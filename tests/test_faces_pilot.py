"""Offline tests for the face pilot. OpenCLIP-specific checks use a tiny random model, never downloaded weights."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_h2
import torch

from mfdime import clip_smile, dime, h2_faces_pilot, setups


class SmileTests(unittest.TestCase):
    def test_original_accuracy_exposes_single_class_predictions(self):
        result = clip_smile.classification_report([0, 0, 0, 1], [0, 0, 0, 0])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertEqual(result["classes"]["1"]["recall"], 0)
        self.assertEqual(result["confusion_true_rows_predicted_columns"], [[3, 0], [1, 0]])
        self.assertIsNone(clip_smile.classification_report([1, 1], [1, 1])["balanced_accuracy"])

    def test_selection_uses_annotations_and_original_predictions_only(self):
        labels = [0, 0, 1, 1, 1, 0]
        selected = h2_faces_pilot.choose_faces(labels, labels, [0, 0, 1, 1, 0, 0], 4)
        self.assertEqual(selected, [0, 1, 2, 3])
        self.assertEqual(h2_faces_pilot.choose_faces([0, 1], [1, 1], [1, 0], 4), [])

    @unittest.skipUnless(importlib.util.find_spec("open_clip"), "install the optional faces dependency group")
    def test_real_openclip_api_with_tiny_model_and_local_preprocessing(self):
        open_clip = importlib.import_module("open_clip")
        clip_model = importlib.import_module("open_clip.model")
        image_transform = importlib.import_module("open_clip.transform").image_transform

        model = clip_model.CLIP(embed_dim=8,
                                vision_cfg=clip_model.CLIPVisionCfg(layers=1, width=32, head_width=8, patch_size=8, image_size=32),
                                text_cfg=clip_model.CLIPTextCfg(context_length=77, vocab_size=49408, width=32, heads=4, layers=1))
        preprocess = image_transform(32, is_train=False, mean=(0.5,) * 3, std=(0.5,) * 3)
        evaluator = clip_smile.SmileEvaluator(model, preprocess, open_clip.get_tokenizer(clip_smile.MODEL)(list(clip_smile.PROMPTS)),
                                               "cpu")
        images = torch.randn(2, 3, 40, 40).clamp(-1, 1).requires_grad_()
        actual = evaluator(images)
        with torch.no_grad():
            transformed = torch.stack([preprocess(clip_smile.to_pil(x)) for x in images])
            features = model.encode_image(transformed, normalize=True)
            expected = features @ evaluator.text_features.T
        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(actual.shape), (2, 2))
        self.assertFalse(actual.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in evaluator.parameters()))
        self.assertIsNone(images.grad)
        # A local checkpoint must use the same architecture/preprocessing as the pinned model.
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / clip_smile.WEIGHT_FILE).touch()
            (folder / "open_clip_config.json").write_text(json.dumps({"model_cfg": open_clip.get_model_config(clip_smile.MODEL),
                                                                    "preprocess_cfg": {"mean": [0.5] * 3, "std": [0.5] * 3}}))
            (folder / "download.json").write_text(json.dumps({"repo": clip_smile.REPO, "revision": clip_smile.REVISION}))
            with patch.object(open_clip, "create_model_and_transforms", return_value=(model, None, preprocess)) as build:
                loaded, info = clip_smile.load(folder, "cpu")
            self.assertEqual(build.call_args.kwargs["pretrained"], str(folder / clip_smile.WEIGHT_FILE))
            self.assertEqual(build.call_args.kwargs["image_mean"], (0.5, 0.5, 0.5))
            self.assertEqual(info["revision"], clip_smile.REVISION)
            self.assertFalse(info["scores_are_probabilities"])
            torch.testing.assert_close(loaded(images), expected)

    def test_face_pilot_end_to_end_and_correct_binary_source_labels(self):
        s = test_h2.PilotTests().setup_toy()
        s.dataset = "celebahq"
        x = torch.tensor([0.1, -0.1, 0.2, -0.2]).view(4, 1, 1, 1).expand(-1, 3, 8, 8).clone()
        labels = [0, 1, 0, 1]
        ph = setups.Photos([f"{i}.jpg" for i in range(4)], x, torch.tensor([1, 0, 1, 0]))
        record = {"label": labels, "guide_pred": labels, "names": ph.names, "split": "practice"}
        info = {"prompts": list(clip_smile.PROMPTS), "model": "toy"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pilot"
            with patch.object(h2_faces_pilot, "repo_sha", return_value=h2_faces_pilot.pmf.FIRE_SHA), \
                 patch.object(setups, "build", return_value=s), \
                 patch.object(setups, "photos", return_value=(ph, record)), \
                 patch.object(clip_smile, "load", return_value=(test_h2.Classifier(), info)), \
                 patch.object(torch.cuda, "is_available", return_value=False), \
                 patch.object(dime, "dime_schedule", return_value=dime.Schedule(0.4, [0.3, 0.15, 0.0])), \
                 contextlib.redirect_stdout(io.StringIO()):
                h2_faces_pilot.main(["--out-dir", str(root)])
                with self.assertRaises(SystemExit):
                    h2_faces_pilot.main(["--out-dir", str(root)])
            self.assertTrue((root / "complete.json").is_file())
            self.assertEqual(json.loads((root / "evaluator.json").read_text())["openclip"]["n"], 4)
            self.assertTrue((root / "originals.csv").is_file())
            for peek in ("jump1", "jump4"):
                for mode in ("shortcut", "exact"):
                    folder = root / peek / mode
                    rows = [json.loads(line) for line in (folder / "manifest.jsonl").read_text().splitlines()]
                    self.assertEqual(len(rows), 4)
                    self.assertEqual([r["source"] for r in rows], labels)
                    self.assertTrue(all(r["source"] == 1 - r["target"] and r["dataset"] == "celebahq" for r in rows))
                    self.assertTrue((folder / "grid_00.jpg").is_file())
                    self.assertTrue(all("smile_margin_after" in r for r in rows))


if __name__ == "__main__":
    unittest.main()
