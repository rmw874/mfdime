"""CPU checks for H2's gradients, sampling pairing, cohort boundary, and four-arm pilot plumbing.

PYTHONPATH=src python -m unittest discover -s tests -v
No weights, datasets, submodules, or network are required.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import torch
from torch import nn

from mfdime import dime, flow, h2_pilot, score, setups


class Classifier(nn.Module):
    def forward(self, x):
        value = x.flatten(1).mean(1)
        return torch.stack([value, -value], dim=1)


class Distance(dime.Perceptual):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x0, x1):
        return (x0 - x1).square().flatten(1).mean(1).sum()


class ToyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.2, 0.7, 1.1]).view(1, 3, 1, 1), requires_grad=False)

    def forward(self, x, t, r, label=None):
        return x * self.weight + 0.01


class IdentitySpace(nn.Module):
    def encode(self, x):
        return x

    def decode(self, x):
        return x


class GradientTests(unittest.TestCase):
    def test_shortcut_matches_pre_h2_fixture(self):
        # Captured from the original committed DiME loop before adding H2, including a scale retry.
        x = torch.tensor([[0.1, 0.2, 0.3], [-0.1, 0.1, -0.2]]).reshape(2, 3, 1, 1)
        grid = [0.4, 0.2, 0.1, 0.0]
        result = dime.explain(lambda x, t: x * 0.2, dime.jump_peek(lambda x, t, r: x * 0.3 + 0.01, grid, 1),
                              lambda z: z, Classifier(), Distance(), x, x, x * 0.7, torch.tensor([1, 0]),
                              grid, 1.0, ["a:seed0", "b:seed0"])
        expected = torch.tensor([[-0.3659955561, -0.0910756588, 0.0218959004],
                                 [0.0227247383, 0.2187804580, 0.3948448300]]).reshape_as(x)
        torch.testing.assert_close(result[0], expected)
        self.assertEqual(result[1].tolist(), [0, 1])
        self.assertEqual(result[2].tolist(), [5, 10])

    def test_composed_jump_gradient_matches_finite_difference_and_product(self):
        for n in (1, 4):
            with self.subTest(jumps=n):
                x = torch.tensor([0.15, -0.2, 0.3], dtype=torch.float64).reshape(1, 3, 1, 1).requires_grad_()
                a = torch.tensor([0.2, 0.7, 1.1], dtype=torch.float64).reshape_as(x)

                def fn(z, a=a, n=n):
                    return flow.jump(lambda z, t, r: a * z + 0.01, z, 0.4, 0.0, n)

                self.assertTrue(torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-7))
                derivative = torch.autograd.grad(fn(x).sum(), x)[0]
                torch.testing.assert_close(derivative, (1 - 0.4 * a / n).pow(n))

    def test_exact_guidance_matches_loss_finite_difference(self):
        # Exercise the real guided loop, including decoding, both loss terms, and the sampler conversion.
        for n in (1, 4):
            with self.subTest(jumps=n):
                captures = []
                grid = [0.4, 0.2, 0.0]
                a = torch.tensor([0.2, 0.7, 1.1], dtype=torch.float64).reshape(1, 3, 1, 1)
                original = torch.full_like(a, 0.1)
                target = torch.tensor([1])
                scale = 8.0

                def u(x, t, r, a=a):
                    return a * x + 0.01

                def fn(z, k, unused, captures=captures, grid=grid, n=n, u=u):
                    z.register_hook(lambda g: captures.append((z.detach().clone(), g.detach().clone())))
                    return flow.jump(u, z, grid[k], 0.0, n)

                def loss(z, grid=grid, n=n, scale=scale, target=target, original=original, u=u):
                    image = 0.5 * flow.jump(u, z, grid[1], 0.0, n)
                    return scale * dime.class_loss(Classifier()(image), target, False) + Distance()(image, original)

                trace = []
                start = original.clone().requires_grad_()
                out, calls = dime.guided_run(lambda z, t: torch.zeros_like(z), dime.Peek(fn, lambda k, n=n: n, True),
                                            lambda x: 0.5 * x, Classifier(), Distance(), original, original, start,
                                            target, scale, grid, [torch.Generator().manual_seed(7)],
                                            gradient_mode="exact", trace=trace.append)
                z, actual = captures[0]
                numerical = torch.zeros_like(z)
                for i in range(z.numel()):
                    delta = torch.zeros_like(z)
                    delta.flatten()[i] = 1e-6
                    numerical.flatten()[i] = (loss(z + delta) - loss(z - delta)) / 2e-6
                torch.testing.assert_close(actual, numerical, atol=1e-7, rtol=1e-6)
                c_t, c_s = dime.vp_scale(grid[1]), dime.vp_scale(grid[2])
                grad = (numerical * c_t + dime.L1 * torch.sign(z / c_t - original)) / c_s
                self.assertAlmostEqual(trace[1]["guidance_l2"][0], float(grad.norm()), places=6)
                self.assertEqual([r["full_gradient"] for r in trace], [False, True])
                self.assertEqual(calls, 2 + n)
                self.assertFalse(out.requires_grad)
                self.assertIsNone(start.grad)

    def test_first_step_identical_and_noise_is_batch_independent(self):
        original = torch.tensor([0.1, -0.1]).reshape(2, 1, 1, 1).expand(-1, 3, 2, 2).clone()
        keys = ["photo_a:seed0", "photo_b:seed0"]
        target = torch.ones(2, dtype=torch.long)
        def v(x, t):
            return x * 0.2

        def u(x, t, r):
            return x * 0.3
        for n in (1, 4):
            for mode in ("shortcut", "exact"):
                def run(x, names, steps, n=n, mode=mode):
                    trace = []
                    result = dime.explain(v, dime.jump_peek(u, steps, n), lambda z: z, Classifier(), Distance(),
                                          x, x, x * 0.7, target[:len(x)], steps, 1.0, names,
                                          gradient_mode=mode, trace=trace.append)
                    return result, trace

                batched, trace = run(original, keys, [0.4, 0.2, 0.0])
                individual = [run(original[i:i + 1], keys[i:i + 1], [0.4, 0.2, 0.0])[0] for i in range(2)]
                for j in range(3):
                    torch.testing.assert_close(batched[j], torch.cat([r[j] for r in individual]))
                self.assertTrue(all(row["finite"] for row in trace))
                self.assertFalse(trace[0]["full_gradient"])
        first = []
        for mode in ("shortcut", "exact"):
            first.append(dime.guided_run(v, dime.jump_peek(u, [0.4, 0.0], 1), lambda z: z, Classifier(), Distance(),
                                        original, original, original * 0.7, target, 8.0, [0.4, 0.0],
                                        [torch.Generator().manual_seed(i) for i in range(2)], gradient_mode=mode)[0])
        torch.testing.assert_close(*first, rtol=0, atol=0)

    def test_exact_rejects_walk_and_nonfinite_gradients(self):
        x = torch.ones(1, 3, 1, 1)
        with self.assertRaisesRegex(ValueError, "jump"):
            dime.guided_run(None, dime.Peek(lambda z, k, v: z, lambda k: 0), None, None, Distance(),
                            x, x, x, torch.tensor([1]), 8.0, [0.4, 0.0], [], gradient_mode="exact")
        with self.assertRaises(FloatingPointError):
            dime.guided_run(lambda z, t: z, dime.jump_peek(lambda z, t, r: z, [0.4, 0.0], 1),
                            lambda z: z * float("nan"), Classifier(), Distance(), x, x, x, torch.tensor([1]),
                            8.0, [0.4, 0.0], [], trace=lambda row: None)


class PilotTests(unittest.TestCase):
    def setup_toy(self):
        net = ToyFlow()
        return setups.Setup("imagenet", setups.Twins(net, net), cast(setups.dmf.Autoencoder, IdentitySpace()), Classifier(), Classifier(),
                            Distance(), False, 0, {"test": "tiny frozen model"})

    def test_cohort_filters_inside_existing_split(self):
        args = argparse.Namespace(n=2, batch=1, device="cpu", source=0, smoke=False)
        pool = setups.Photos(["a", "b", "c"], torch.zeros(3, 3, 2, 2), torch.ones(3, dtype=torch.long))
        with patch.object(setups, "photos", return_value=(pool, {})) as load, \
             patch.object(h2_pilot, "predict", return_value=torch.tensor([1, 0, 0])):
            chosen, record = h2_pilot.cohort(args, self.setup_toy())
        self.assertEqual(load.call_args.args[2:], ("practice", 0))
        self.assertEqual(chosen.names, ["b", "c"])
        self.assertEqual(record["eligible"], 2)

    def test_four_arm_pilot_outputs_and_refuses_overwrite(self):
        s = self.setup_toy()
        x = torch.full((2, 3, 8, 8), 0.1)
        ph = setups.Photos(["a.jpg", "b.jpg"], x, torch.ones(2, dtype=torch.long))
        record = {"names": ph.names, "eligible": 2}
        sched = dime.Schedule(0.4, [0.35, 0.2, 0.1, 0.0])

        def prepare(args):
            if args.out_dir.exists():
                raise SystemExit("existing output")
            args.out_dir.mkdir()
            args.device, args.gain = "cpu", 1.0
            return s, sched

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pilot"
            argv = ["--source", "0", "--target", "1", "--out-dir", str(root), "--n", "2"]
            with patch.object(h2_pilot.explain, "prepare", side_effect=prepare), \
                 patch.object(h2_pilot, "cohort", return_value=(ph, record)), \
                 patch.object(h2_pilot, "repo_sha", return_value=h2_pilot.dmf.DMF_SHA):
                h2_pilot.main(argv)
                with self.assertRaises(SystemExit):
                    h2_pilot.main(argv)
            self.assertTrue((root / "complete.json").exists())
            report = json.loads((root / "pilot.json").read_text())
            self.assertEqual(len(report["arms"]), 4)
            self.assertEqual(len(report["comparisons"]), 2)
            for peek, mode in h2_pilot.ARMS:
                arm = root / peek / mode
                rows = [json.loads(line) for line in (arm / "manifest.jsonl").read_text().splitlines()]
                trace = [json.loads(line) for line in (arm / "diagnostics.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(r["gradient_mode"] == mode and r["check_before"] == 0 for r in rows))
                self.assertEqual(sum(r["full_gradient"] for r in trace) > 0, mode == "exact")
                self.assertTrue((arm / "grid_00.jpg").exists())
            self.assertTrue(all(p.grad is None for p in s.twins.flow_map.parameters()))

    def test_h1_scorer_rejects_exact_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.jsonl").write_text(json.dumps({"gradient_mode": "exact"}) + "\n")
            with self.assertRaisesRegex(SystemExit, "H1"):
                score.load_runs(root)


if __name__ == "__main__":
    unittest.main()
