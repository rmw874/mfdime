"""Full faces checks without real weights or data: frozen cohort, five seeds, sample resume and scoring."""

from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_h2
import torch

from mfdime import celebahq, clip_smile, config, dime, h2, h2_faces, h2_faces_score, setups


def fake_images(path, idxs):
    return torch.stack([torch.full((3, 8, 8), 0.1 if i % 2 == 0 else -0.1) for i in idxs])


def fake_faces():
    return [celebahq.Face(i, {"smile": i % 2}) for i in range(8)]


class FullFaceTests(unittest.TestCase):
    def test_cohort_excludes_practice_and_tuning_and_freezes_labels(self):
        s = test_h2.PilotTests().setup_toy()
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(out_dir=Path(tmp), celebahq_zip=Path("unused"), celeba_partition=Path("unused"),
                                      device="cpu", n=2)
            with patch.object(celebahq, "load_split", side_effect=lambda *args: fake_faces()), \
                 patch.object(celebahq, "load_images", side_effect=fake_images), \
                 patch.object(config, "N_PRACTICE", 2), patch.object(h2_faces, "SKIP_TUNING", 1):
                record = h2_faces.frozen_cohort(args, s, test_h2.Classifier(), "signature")
                self.assertEqual(record["excluded_names"], ["0.jpg", "1.jpg", "2.jpg"])
                self.assertEqual(record["names"], ["3.jpg", "4.jpg"])
                self.assertEqual([r["target"] for r in record["selected"]], [0, 1])
                with patch.object(celebahq, "load_images", side_effect=AssertionError("cohort was reselected")):
                    self.assertEqual(h2_faces.frozen_cohort(args, s, None, "signature"), record)
            changed = dict(record, names=["0.jpg", "4.jpg"])
            changed["cohort_sha256"] = h2.digest({k: v for k, v in changed.items() if k != "cohort_sha256"})
            with self.assertRaises(ValueError):
                h2_faces.validate_cohort(changed, "signature")

    def test_all_five_seeds_end_to_end_resume_and_direction_score(self):
        s = test_h2.PilotTests().setup_toy()
        s.dataset, s.use_logits = "celebahq", True
        sched = dime.Schedule(0.4, [0.3, 0.15, 0.0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "faces"
            with patch.object(celebahq, "load_split", side_effect=lambda *args: fake_faces()), \
                 patch.object(celebahq, "load_images", side_effect=fake_images), \
                 patch.object(config, "N_PRACTICE", 2), patch.object(h2_faces, "SKIP_TUNING", 1), \
                 patch.object(h2_faces, "repo_sha", return_value=h2_faces.pmf.FIRE_SHA), \
                 patch.object(setups, "build", return_value=s) as build, \
                 patch.object(clip_smile, "load", return_value=(test_h2.Classifier(), {"model": "test"})), \
                 patch.object(torch.cuda, "is_available", return_value=False), \
                 patch.object(dime, "dime_schedule", return_value=sched), contextlib.redirect_stdout(io.StringIO()):
                for seed in range(5):
                    h2_faces.main(["--seed", str(seed), "--n", "2", "--out-dir", str(root)])
                self.assertEqual(build.call_count, 5)
                records = [(p, p.read_bytes()) for p in root.glob("runs/**/samples/*/complete.json")]
                self.assertEqual(len(records), 2 * 4 * 5)
                with patch.object(h2_faces.explain, "counterfactuals", side_effect=AssertionError("completed edit rerun")):
                    h2_faces.main(["--seed", "0", "--n", "2", "--out-dir", str(root)])
                self.assertTrue(all(p.read_bytes() == content for p, content in records))
                protocol, cohorts, results = h2_faces_score.load_results(root)
                self.assertEqual(protocol["batch"], 1)
                self.assertEqual(cohorts["add_smile"]["names"], ["4.jpg"])
                self.assertEqual(cohorts["remove_smile"]["names"], ["3.jpg"])
                score = h2_faces_score.analyse(protocol, cohorts, results, 100, 0)
                self.assertEqual(score["photos"], 2)
                self.assertTrue(all(v["explanations"] == 10 for v in score["arms"].values()))
                self.assertEqual(len(score["directions"]), 2)
                h2_faces_score.main(["--runs", str(root), "--out-dir", str(root / "score"), "--boot", "100"])
                self.assertTrue((root / "score/score.json").exists())
                self.assertIn("OpenCLIP", (root / "score/score.md").read_text())
                with self.assertRaises(ValueError):
                    h2_faces.main(["--seed", "0", "--n", "3", "--out-dir", str(root)])
                # Partial copying is rejected even if arm and seed markers were copied.
                records[0][0].unlink()
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    h2_faces_score.load_results(root)

    def test_numeric_failures_are_checkpointed_without_scoring_or_retry(self):
        s = test_h2.PilotTests().setup_toy()
        args = argparse.Namespace(peek="jump4", gradient_mode="exact", gain=3.0, device="cpu", celebahq_zip=Path("unused"))
        entry = {"photo": "4.jpg", "idx": 4, "label": 0, "target": 1, "guide_before": 0,
                 "check_before": 0, "smile_margin_before": -0.1}
        identity = {"photo": "4.jpg", "seed": 0, "signature": "test"}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "samples/face"
            with patch.object(celebahq, "load_images", side_effect=fake_images), \
                 patch.object(h2_faces.explain, "counterfactuals", side_effect=FloatingPointError("nonfinite")) as run:
                row = h2_faces.run_sample(args, s, None, entry, 0, dime.Schedule(0.4, [0.3, 0.0]), folder, identity)
                self.assertFalse(row["check_flipped"])
                self.assertEqual(row["status"], "numerical_failure")
                self.assertIsNone(row["target_margin_change"])
                self.assertEqual(h2_faces.run_sample(args, s, None, entry, 0, dime.Schedule(0.4, [0.3, 0.0]), folder, identity), row)
                self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
