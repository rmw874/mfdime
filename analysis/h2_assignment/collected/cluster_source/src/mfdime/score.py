"""Score H1 from saved runs: with DiME's shortcut, does one jump (jump1) give a near-identical flip rate and sFID to
the noise-free walk (euler_rest) over 5 seeds, at about 14x fewer generator calls?

  python -P -m mfdime.score --runs results/h1_celebahq_gain3_v1 --out-dir results/h1_celebahq_gain3_v1/score
  python -P -m mfdime.score --runs results/h1_imagenet_v1 --out-dir results/h1_imagenet_v1/score

Reads every manifest.jsonl under --runs (mfdime.explain runs, any folder layout), pairs the two peeks by photo and seed,
and reports per peek: flip rate, FID, sFID, generator calls and seconds per explanation (the GPU each run had), and
the flip rate per target class (on faces: towards and away from the attribute).

  FID   ACE's compute_FID.sh: pytorch_fid's Inception features (2048-d pool), the originals against the counterfactuals
        that flipped, one value per seed.
  sFID  ACE's compute_sFID.sh and Sec. 4.1: the photos split at random into two halves, the FID between one half's
        originals and the other half's flipped counterfactuals, both ways, over 10 splits (the same splits for every
        peek and seed), averaged; one value per seed.

H1's rule, fixed before the runs: for flip rate and sFID, the difference jump1 - euler_rest (mean over seeds) is
near-identical when its 95% bootstrap interval (photos resampled with replacement, the same draw for both peeks and
every seed) lies within plus or minus the noise floor, euler_rest's standard deviation over the seeds.

Writes score.json and score.md to --out-dir (refuses to overwrite). --smoke scores tiny runs with random Inception
weights (plumbing only, no download).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

PEEKS = ("jump1", "euler_rest")
REFERENCE = "euler_rest"
SPLITS = 10


@dataclass
class Arm:
    """One peek's explanations, as arrays over (seed, photo) in the order of Runs.seeds and Runs.photos."""

    flipped: np.ndarray  # bool
    calls: np.ndarray
    seconds: np.ndarray
    cf: list[list[Path]]  # counterfactual image files
    feats: np.ndarray | None = None  # (seeds, photos, d) Inception features


@dataclass
class Runs:
    photos: list[str]
    seeds: list[int]
    target: np.ndarray  # (photos,) target class
    originals: list[Path]
    arms: dict[str, Arm]
    orig_feats: np.ndarray | None = None


def load_runs(root: Path) -> Runs:
    """Every manifest row under root, grouped by peek; the two peeks must cover the same photos and seeds."""
    rows: dict[str, dict[tuple[str, int], tuple[dict, Path]]] = {p: {} for p in PEEKS}
    for manifest in sorted(root.rglob("manifest.jsonl")):
        for line in manifest.read_text().splitlines():
            r = json.loads(line)
            if r.get("gradient_mode", "shortcut") != "shortcut":
                raise SystemExit("STOP: mfdime.score is the H1 shortcut-only scorer; use h2_pilot's pilot report for H2")
            if r["peek"] in rows:
                key = (r["photo"], r["seed"])
                if key in rows[r["peek"]]:
                    raise SystemExit(f"STOP: {r['peek']} has photo {key[0]} seed {key[1]} twice under {root}")
                rows[r["peek"]][key] = (r, manifest.parent)
    keys = {p: set(rows[p]) for p in PEEKS}
    if not keys[PEEKS[0]] or keys[PEEKS[0]] != keys[PEEKS[1]]:
        raise SystemExit(f"STOP: under {root} the peeks do not cover the same photos and seeds "
                         f"({', '.join(f'{p}: {len(keys[p])} explanations' for p in PEEKS)})")
    photos = sorted({ph for ph, _ in keys[PEEKS[0]]})
    seeds = sorted({s for _, s in keys[PEEKS[0]]})
    if len(keys[PEEKS[0]]) != len(photos) * len(seeds):
        raise SystemExit(f"STOP: not every photo has every seed under {root}")
    any_rows = rows[PEEKS[0]]
    target = np.array([any_rows[(ph, seeds[0])][0]["target"] for ph in photos])
    originals = [any_rows[(ph, seeds[0])][1] / "images" / f"{Path(ph).stem}_orig.png" for ph in photos]
    arms = {}
    for p in PEEKS:
        cell = [[rows[p][(ph, s)] for ph in photos] for s in seeds]
        arms[p] = Arm(flipped=np.array([[r["flipped"] for r, _ in row] for row in cell]),
                      calls=np.array([[r["generator_calls"] for r, _ in row] for row in cell], dtype=float),
                      seconds=np.array([[r["seconds"] for r, _ in row] for row in cell], dtype=float),
                      cf=[[d / "images" / f"{Path(r['photo']).stem}_seed{r['seed']}_cf.png" for r, d in row] for row in cell])
    return Runs(photos, seeds, target, originals, arms)


class SmokeFeatures(nn.Module):
    """A fixed random stand-in for Inception (64-d), so the plumbing runs without the FID weights."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(3, 64, 7, stride=4)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [self.conv(x).relu().mean((2, 3), keepdim=True)]


def feature_net(smoke: bool, device: str) -> nn.Module:
    if smoke:
        return SmokeFeatures().eval().to(device)
    from pytorch_fid.inception import InceptionV3

    try:  # pytorch_fid's FID Inception weights, from the torch hub cache
        net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
    except OSError as e:
        raise SystemExit(f"STOP: pytorch_fid's Inception weights not in the torch hub cache ({e})") from e
    return net.eval().to(device)


@torch.no_grad()
def features(net: nn.Module, paths: list[Path], device: str, batch: int) -> np.ndarray:
    """Features of image files read as pytorch_fid reads them (RGB, scaled to [0, 1])."""
    out = []
    for i in range(0, len(paths), batch):
        imgs = [torch.from_numpy(np.array(Image.open(p).convert("RGB"))).permute(2, 0, 1) for p in paths[i:i + batch]]
        x = torch.stack(imgs).float().div(255).to(device)
        out.append(net(x)[0].flatten(1).double().cpu().numpy())
    return np.concatenate(out)


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    """The Frechet distance between Gaussians fitted to two feature sets (rows = samples). The trace of the matrix
    square root of S_a S_b is computed as the nuclear norm of the centred cross product A B^T / sqrt((n_a - 1)(n_b - 1)),
    which has the same nonzero eigenvalues, so it is exact without a d x d square root. NaN if a set has fewer than two
    samples."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ca, cb = a - a.mean(0), b - b.mean(0)
    na, nb = len(a) - 1, len(b) - 1
    cross = np.linalg.svd(ca @ cb.T / np.sqrt(na * nb), compute_uv=False).sum()
    mean_term = float(((a.mean(0) - b.mean(0)) ** 2).sum())
    return mean_term + float((ca ** 2).sum() / na + (cb ** 2).sum() / nb - 2 * cross)


def nanmean(v) -> float:
    """The mean of the finite values; NaN (without a warning) when there are none."""
    v = np.asarray(v, dtype=float)
    return float(v[np.isfinite(v)].mean()) if np.isfinite(v).any() else float("nan")


def nansd(v) -> float:
    """The standard deviation (ddof 1) of the finite values; NaN when there are fewer than two."""
    v = np.asarray(v, dtype=float)
    return float(v[np.isfinite(v)].std(ddof=1)) if np.isfinite(v).sum() > 1 else float("nan")


def splits(n: int, count: int = SPLITS, seed: int = 0) -> list[np.ndarray]:
    """count random orders of n positions; the first half of each order is one half of the split."""
    rng = np.random.default_rng(seed)
    return [rng.permutation(n) for _ in range(count)]


def fid_per_seed(runs: Runs, arm: Arm, idx: np.ndarray) -> np.ndarray:
    """FID per seed on the photos idx (positions into runs.photos, repeats allowed)."""
    assert runs.orig_feats is not None and arm.feats is not None
    real = runs.orig_feats[idx]
    return np.array([frechet(real, arm.feats[s][idx][arm.flipped[s][idx]]) for s in range(len(runs.seeds))])


def sfid_per_seed(runs: Runs, arm: Arm, idx: np.ndarray, orders: list[np.ndarray]) -> np.ndarray:
    """sFID per seed on the photos idx: for each order, the halves' originals against the other halves' flipped
    counterfactuals, both ways, averaged over orders."""
    assert runs.orig_feats is not None and arm.feats is not None
    out = []
    for s in range(len(runs.seeds)):
        vals = []
        for order in orders:
            half = len(order) // 2
            h1, h2 = idx[order[:half]], idx[order[half:2 * half]]
            for real, cf in ((h1, h2), (h2, h1)):
                vals.append(frechet(runs.orig_feats[real], arm.feats[s][cf][arm.flipped[s][cf]]))
        out.append(nanmean(vals))
    return np.array(out)


def per_seed(runs: Runs, arm: Arm, idx: np.ndarray, orders: list[np.ndarray]) -> dict[str, np.ndarray]:
    return {"flip_rate": arm.flipped[:, idx].mean(1), "fid": fid_per_seed(runs, arm, idx),
            "sfid": sfid_per_seed(runs, arm, idx, orders), "generator_calls": arm.calls[:, idx].mean(1),
            "seconds": arm.seconds[:, idx].mean(1)}


def score(runs: Runs, boot: int, seed: int) -> dict:
    n = len(runs.photos)
    everyone = np.arange(n)
    orders = splits(n, seed=seed)
    result: dict = {"photos": n, "seeds": runs.seeds, "peeks": {}, "h1": {}}
    observed = {}
    for p in PEEKS:
        vals = per_seed(runs, runs.arms[p], everyone, orders)
        observed[p] = vals
        flips_by_target = {str(c): float(runs.arms[p].flipped[:, runs.target == c].mean()) for c in np.unique(runs.target)}
        result["peeks"][p] = {m: {"mean": nanmean(v), "sd_over_seeds": nansd(v),
                                  "per_seed": [float(x) for x in v]} for m, v in vals.items()}
        result["peeks"][p]["flip_rate_by_target"] = flips_by_target
    rng = np.random.default_rng(seed + 1)
    draws: dict[str, list[float]] = {"flip_rate": [], "sfid": []}
    for _ in range(boot):
        idx = rng.integers(0, n, n)
        b_orders = splits(n, seed=int(rng.integers(1 << 31)))
        b = {p: per_seed(runs, runs.arms[p], idx, b_orders) for p in PEEKS}
        for m in draws:
            draws[m].append(nanmean(b[PEEKS[0]][m]) - nanmean(b[REFERENCE][m]))
    for m in draws:
        diff = nanmean(observed[PEEKS[0]][m]) - nanmean(observed[REFERENCE][m])
        finite = np.array([d for d in draws[m] if np.isfinite(d)])
        lo, hi = (float(x) for x in np.percentile(finite, [2.5, 97.5])) if len(finite) else (float("nan"),) * 2
        floor = nansd(observed[REFERENCE][m])
        result["h1"][m] = {"difference": diff, "ci95": [lo, hi], "noise_floor": floor,
                           "near_identical": bool(-floor <= lo and hi <= floor)}
    calls = {p: float(np.mean(runs.arms[p].calls)) for p in PEEKS}
    result["h1"]["call_ratio"] = calls[REFERENCE] / calls[PEEKS[0]]
    result["h1"]["supported"] = all(result["h1"][m]["near_identical"] for m in draws)
    return result


def write_md(args, result: dict) -> None:
    pk, h = result["peeks"], result["h1"]
    def cell(p: str, m: str, f: str) -> str:
        return f"{pk[p][m]['mean']:{f}} ({pk[p][m]['sd_over_seeds']:{f}})"
    md = [f"# H1 score: {args.runs}", "",
          f"{result['photos']} photos x {len(result['seeds'])} seeds, paired by photo and seed. Mean over seeds "
          f"(standard deviation over seeds).", "",
          "| peek | flip rate | sFID | FID | generator calls | s / explanation |", "|---|---|---|---|---|---|"]
    for p in PEEKS:
        md.append(f"| {p} | {cell(p, 'flip_rate', '.3f')} | {cell(p, 'sfid', '.2f')} | {cell(p, 'fid', '.2f')} | "
                  f"{cell(p, 'generator_calls', '.0f')} | {cell(p, 'seconds', '.2f')} |")
    md += ["", "Flip rate by target class: " + "; ".join(
        f"{p}: " + ", ".join(f"target {c} {v:.3f}" for c, v in pk[p]["flip_rate_by_target"].items()) for p in PEEKS), "",
        f"| {PEEKS[0]} - {REFERENCE} | difference | 95% interval | noise floor | near-identical |", "|---|---|---|---|---|"]
    for m in ("flip_rate", "sfid"):
        e = h[m]
        md.append(f"| {m} | {e['difference']:+.4f} | [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] | "
                  f"{e['noise_floor']:.4f} | {'yes' if e['near_identical'] else 'no'} |")
    md += ["", f"Generator calls: {REFERENCE} uses {h['call_ratio']:.1f}x the calls of {PEEKS[0]}.", "",
           ("H1 supported: both differences lie within the noise floor" if h["supported"] else
            "H1 not supported: at least one difference is not within the noise floor") + f" ({args.boot} bootstrap draws)."]
    (args.out_dir / "score.md").write_text("\n".join(md) + "\n")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, required=True, help="the folder holding the explain runs of both peeks")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--boot", type=int, default=500, help="bootstrap draws over photos")
    p.add_argument("--seed", type=int, default=0, help="seeds the sFID splits and the bootstrap")
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--smoke", action="store_true", help="random feature weights, 20 bootstrap draws (plumbing)")
    args = p.parse_args(argv)
    if args.out_dir.exists():
        raise SystemExit(f"STOP: {args.out_dir} exists (scores are written once; pick a new --out-dir)")
    if args.smoke:
        args.boot = 20
    device = "cuda" if torch.cuda.is_available() and not args.smoke else "cpu"
    runs = load_runs(args.runs)
    print(f"{len(runs.photos)} photos x {len(runs.seeds)} seeds; features on {device}", flush=True)
    net = feature_net(args.smoke, device)
    runs.orig_feats = features(net, runs.originals, device, args.batch)
    for arm in runs.arms.values():
        flat = features(net, [f for row in arm.cf for f in row], device, args.batch)
        arm.feats = flat.reshape(len(runs.seeds), len(runs.photos), -1)
    result = score(runs, args.boot, args.seed)
    args.out_dir.mkdir(parents=True)
    result["runs"], result["boot"], result["device"] = str(args.runs), args.boot, device
    (args.out_dir / "score.json").write_text(json.dumps(result, indent=2))
    write_md(args, result)
    print((args.out_dir / "score.md").read_text())


if __name__ == "__main__":
    sys.exit(main())
