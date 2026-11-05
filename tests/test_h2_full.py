"""Held-out H2 checks: concurrent cohort publication, seed partitioning, resume, failures and paired inference."""

from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import test_h2
import torch

from mfdime import config, dime, h2, h2_score, imagenet, setups


class FullExperimentTests(unittest.TestCase):
    def test_seed_partition_is_complete_and_disjoint(self):
        self.assertEqual(config.H2_SEED_GROUPS, ((0, 1), (2, 3), (4,)))
        flattened = [seed for group in config.H2_SEED_GROUPS for seed in group]
        self.assertEqual(flattened, list(config.SEEDS))

    def test_first_cohort_publisher_wins_without_partial_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shared.json"
            with ThreadPoolExecutor(max_workers=3) as pool:
                records = list(pool.map(lambda value: h2.publish_once(path, {"winner": value}), range(3)))
            self.assertTrue(all(r == records[0] for r in records))
            self.assertEqual(h2.read_json(path), records[0])

    def test_freeze_before_evaluator_filter_and_reuse_without_refilter(self):
        s = test_h2.PilotTests().setup_toy()
        images = [(f"photo{i}.jpg", torch.full((3, 4, 4), 0.1)) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(source=0, target=1, batch=1, device="cpu", out_dir=Path(tmp), imagenet_val=Path("unused"))
            with patch.object(imagenet, "load_classes", return_value={0: images}), \
                 patch.object(config, "N_PRACTICE", 2), \
                 patch.object(h2, "predict", side_effect=[torch.tensor([0, 0, 0, 0, 1]), torch.tensor([1, 0])]) as pred:
                ph, record = h2.frozen_photos(args, s, "signature")
                self.assertEqual(record["practice_names"], ["photo0.jpg", "photo1.jpg"])
                self.assertEqual(record["held_out_names"], ["photo2.jpg", "photo3.jpg"])
                self.assertEqual(ph.names, ["photo3.jpg"])
                again, _ = h2.frozen_photos(args, s, "signature")
                self.assertEqual(again.names, ph.names)
                self.assertEqual(pred.call_count, 2)

    def test_sample_failure_and_interrupted_attempt_are_accounted_for(self):
        s = test_h2.PilotTests().setup_toy()
        x = torch.full((1, 3, 4, 4), 0.1)
        ph = setups.Photos(["photo.jpg"], x, torch.tensor([1]))
        args = argparse.Namespace(source=0, target=1, gain=1.0, peek="jump4", gradient_mode="exact")
        identity = {"photo": ph.names[0], "seed": 0, "signature": "test"}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "arm" / "samples" / "photo"
            folder.mkdir(parents=True)
            (folder / "partial.txt").write_text("interrupted")
            with patch.object(h2.explain, "counterfactuals", side_effect=FloatingPointError("bad gradient")) as run:
                row = h2.run_sample(args, s, ph, 0, dime.Schedule(0.4, [0.3, 0.0]), folder, identity)
                self.assertEqual(row["status"], "numerical_failure")
                self.assertFalse(row["check_flipped"])
                self.assertTrue(list((Path(tmp) / "arm" / "interrupted").glob("*/partial.txt")))
                self.assertEqual(h2.run_sample(args, s, ph, 0, dime.Schedule(0.4, [0.3, 0.0]), folder, identity), row)
                self.assertEqual(run.call_count, 1)
            with self.assertRaises(ValueError):
                h2.completed_sample(folder, identity | {"seed": 1})

    def test_all_groups_end_to_end_resume_and_score(self):
        s = test_h2.PilotTests().setup_toy()
        data = {0: [(f"photo{i}.jpg", torch.full((3, 8, 8), 0.1)) for i in range(4)], 1: []}
        schedule = dime.Schedule(0.4, [0.35, 0.2, 0.1, 0.0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "full"
            with patch.object(config, "H2_PAIRS", ((0, 1),)), patch.object(config, "N_PRACTICE", 2), \
                 patch.object(h2, "repo_sha", return_value=h2.dmf.DMF_SHA), \
                 patch.object(setups, "build", return_value=s) as build, \
                 patch.object(imagenet, "load_classes", return_value=data), \
                 patch.object(dime, "dime_schedule", return_value=schedule), \
                 patch.object(torch.cuda, "is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                for group in range(3):
                    h2.main(["--group", str(group), "--out-dir", str(root)])
                self.assertEqual(build.call_count, 3)  # one load per GPU job, not one per arm or photo
                original = [(p, p.read_bytes()) for p in root.glob("runs/**/samples/*/complete.json")]
                with patch.object(h2.explain, "counterfactuals", side_effect=AssertionError("completed sample was rerun")):
                    h2.main(["--group", "0", "--out-dir", str(root)])
                self.assertTrue(all(p.read_bytes() == content for p, content in original))
                protocol, cohorts, results = h2_score.load_results(root)
                self.assertEqual(len(cohorts["0_1"]["names"]), 2)
                self.assertEqual(len(original), 2 * 5 * 4)
                result = h2_score.analyse(protocol, cohorts, results, 100, 0)
                self.assertEqual(result["photos"], 2)
                self.assertTrue(all(a["explanations"] == 10 for a in result["arms"].values()))
                h2_score.main(["--runs", str(root), "--out-dir", str(root / "score"), "--boot", "100"])
                self.assertTrue((root / "score" / "score.md").exists())
                # Never silently score an incomplete/corrupt sample, even if arm/group markers remain.
                original[0][0].unlink()
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    h2_score.load_results(root)


class BootstrapTests(unittest.TestCase):
    def row(self, photo, seed, success):
        return {"photo": photo, "seed": seed, "flipped": success, "check_flipped": success,
                "status": "ok", "scale": 8.0, "pixel_l1": 0.1, "perceptual": 0.2, "seconds": 1.0,
                "generator_calls": 100, "batch_peak_allocated_bytes": None, "batch_peak_reserved_bytes": None}

    def test_seed_replicates_do_not_shrink_photo_bootstrap(self):
        def run(seeds):
            protocol = {"arms": [list(a) for a in config.H2_ARMS], "seeds": seeds}
            cohorts = {"pair": {"names": ["a", "b"]}}
            results = {"pair": {}}
            for peek, mode in config.H2_ARMS:
                results["pair"][f"{peek}/{mode}"] = [[self.row("a", seed, mode == "exact"),
                                                      self.row("b", seed, False)] for seed in seeds]
            return h2_score.analyse(protocol, cohorts, results, 1000, 7)

        one, five = run([0]), run(list(range(5)))
        for peek in ("jump1", "jump4"):
            self.assertEqual(one["comparisons"][peek], five["comparisons"][peek])
            self.assertEqual(five["comparisons"][peek]["difference_exact_minus_shortcut"], 0.5)
            self.assertEqual(five["comparisons"][peek]["ci95"], [0.0, 1.0])

    def test_stratification_preserves_pair_weights(self):
        protocol = {"arms": [list(a) for a in config.H2_ARMS], "seeds": [0]}
        cohorts = {"small": {"names": ["a"]}, "large": {"names": ["b", "c", "d"]}}
        results = {}
        for pair, cohort in cohorts.items():
            results[pair] = {f"{peek}/{mode}": [[self.row(photo, 0, pair == "small" and mode == "exact")
                                                 for photo in cohort["names"]]] for peek, mode in config.H2_ARMS}
        result = h2_score.analyse(protocol, cohorts, results, 100, 0)
        self.assertEqual(result["comparisons"]["jump1"]["ci95"], [0.25, 0.25])
        self.assertAlmostEqual(result["comparisons"]["jump1"]["difference_exact_minus_shortcut"], 0.25)

    def test_numerical_failure_remains_in_denominator(self):
        rows = [self.row("a", 0, True), self.row("b", 0, False)]
        rows[1].update(status="numerical_failure", pixel_l1=None)
        summary = h2_score.arm_summary(rows, [0])
        self.assertEqual(summary["check_success"], 0.5)
        self.assertEqual(summary["numerical_failures"], 1)
        self.assertEqual(summary["pixel_l1"]["observations"], 1)
        self.assertTrue(np.isfinite(summary["pixel_l1"]["mean"]))


if __name__ == "__main__":
    unittest.main()
