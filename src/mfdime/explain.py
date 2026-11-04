"""Make counterfactual explanations and save them for evaluation. DiME's loop as published (mfdime.dime) with a chosen
peek; no metrics beyond what the guided classifier now says, which is what DiME's own runs record.

  python -P -m mfdime.explain --dataset imagenet --peek jump1 --out-dir results/explain/<name>
  python -P -m mfdime.explain --dataset celebahq --peek euler_rest --out-dir ...

Choices (defaults in mfdime.config: 5 seeds, 16 photos, the dataset's gain):
  --peek      walk_rest (DiME's own: the remaining steps with fresh noise, 1830 generator calls), euler_rest (the
              same steps without noise, the reference) or jump<N> (N flow-map jumps: DMF on imagenet, pMF on
              celebahq)
  --gain      multiplies DiME's scales 8/10/15; 0 switches the classifier's nudge off (a control)
  --seeds     one explanation per seed and photo
  --split     practice (the first 16 photos, where the gain is chosen) or held_out (the photos after them; on
              celebahq the rest of CelebA's test partition)
  --skip, --n leave out the split's first SKIP photos, then explain N of the rest (--n 0: all of them)

Output in --out-dir: images/<photo>_orig.png and images/<photo>_seed<k>_cf.png (256 px), manifest.jsonl (one row per
explanation: photo, seed, peek, gain, target, the guide's answer before and after, flipped, the scale it took,
generator calls, seconds), photos.json (how the photos were chosen), config.json, env.json and grid.jpg (originals,
then each seed's counterfactual).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from mfdime import celebahq, config, dime, imagenet, setups
from mfdime.common import batches, env_info, key_seed, predict, save_grid, save_png


def prepare(args) -> tuple[setups.Setup, dime.Schedule]:
    """Resolve the defaults, create --out-dir, build the setup and record env.json. Returns the setup and DiME's
    schedule."""
    sched = dime.dime_schedule()
    if args.gain is None:
        args.gain = config.GAIN[args.dataset]
    if args.smoke:
        args.n, args.batch, sched = 2, 2, dime.dime_schedule(start_step=3, respacing=10, diffusion_steps=20)
    args.device = "cuda" if torch.cuda.is_available() and not args.smoke else "cpu"
    if args.out_dir.exists():
        raise SystemExit(f"STOP: {args.out_dir} exists (runs are written once; pick a new --out-dir)")
    args.out_dir.mkdir(parents=True)
    if args.smoke and args.dataset == "imagenet":
        args.imagenet_val = args.out_dir / "fake_val"
        imagenet.fake_shard(args.imagenet_val / "validation-00000-of-00001.parquet", [args.source] * 3 + [args.target] * 2)
    if args.smoke and args.dataset == "celebahq":
        args.celebahq_zip, args.celeba_partition = celebahq.fake_dataset(args.out_dir / "fake_celebahq",
                                                                         n=config.N_PRACTICE + args.n)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    print("load", flush=True)
    s = setups.build(args)
    (args.out_dir / "env.json").write_text(json.dumps(env_info(args, s.info), indent=2))
    return s, sched


def counterfactuals(args, s: setups.Setup, ph: setups.Photos, arm: str, seeds: list[int], sched: dime.Schedule,
                    out_dir: Path) -> tuple[list[dict], list[Tensor]]:
    """DiME with one peek for every photo and seed. Writes images/<photo>_seed<k>_cf.png and manifest.jsonl to out_dir;
    returns the manifest rows and the counterfactuals, one (N, 3, 256, 256) tensor per seed. Each photo's starting
    noise and step noise depend only on the photo and the seed, so every arm sees the same noise whatever the batch or
    the set of photos."""
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    v, u = s.twins.v(s.null_label), s.twins.u(s.null_label)
    with torch.no_grad():
        lat = torch.cat([s.ae.encode(ph.x[b].to(args.device)) for b in batches(len(ph.x), args.batch)])
    before = predict(s.guide, ph.x, args.batch, args.device)
    rows, per_seed = [], []
    with (out_dir / "manifest.jsonl").open("w") as manifest:
        for seed in seeds:
            print(f"{arm} seed {seed}", flush=True)
            keys = [f"{name}:seed{seed}" for name in ph.names]
            eps = torch.stack([torch.randn(lat.shape[1:], generator=torch.Generator().manual_seed(key_seed(f"{key}:start")))
                               for key in keys]).to(args.device)
            z_start = dime.start_latent(lat, eps, sched)
            peek = dime.peek(arm, v, u, sched.grid, torch.Generator().manual_seed(seed + 100))
            out = []
            for sl in batches(len(ph.x), args.batch):
                if args.device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                z, used, calls = dime.explain(v, peek, s.ae.decode, s.guide, s.perc, ph.x[sl].to(args.device), lat[sl],
                                              z_start[sl], ph.target[sl], sched.grid, args.gain, keys[sl],
                                              use_logits=s.use_logits)
                with torch.no_grad():
                    img = s.ae.decode(z).clamp(-1, 1).cpu()
                if args.device == "cuda":
                    torch.cuda.synchronize()
                secs = (time.perf_counter() - start) / len(img)
                after = predict(s.guide, img, args.batch, args.device)
                for j, i in enumerate(range(sl.start, sl.stop)):
                    save_png(img[j], out_dir / "images" / f"{Path(ph.names[i]).stem}_seed{seed}_cf.png")
                    row = {"photo": ph.names[i], "seed": seed, "peek": arm, "gain": args.gain,
                           "target": int(ph.target[i]), "guide_before": int(before[i]), "guide_after": int(after[j]),
                           "flipped": bool(after[j] == ph.target[i]),
                           "scale": dime.SCALES[used[j]] if used[j] >= 0 else None,
                           "generator_calls": int(calls[j]), "seconds": round(secs, 3)}
                    manifest.write(json.dumps(row) + "\n")
                    rows.append(row)
                out.append(img)
            per_seed.append(torch.cat(out))
    return rows, per_seed


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    setups.add_args(p)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--peek", default=config.PEEK)
    p.add_argument("--gain", type=float, default=None, help="default: mfdime.config.GAIN for the dataset")
    p.add_argument("--seeds", type=int, nargs="+", default=list(config.SEEDS))
    p.add_argument("--split", choices=("practice", "held_out"), default="practice")
    p.add_argument("--skip", type=int, default=0, help="leave out the split's first SKIP photos")
    p.add_argument("--n", type=int, default=config.N_PHOTOS, help="photos to explain (0: every photo of the split)")
    p.add_argument("--batch", type=int, default=config.BATCH)
    p.add_argument("--smoke", action="store_true", help="CPU, tiny random models, fake data, 3 DiME steps (plumbing)")
    args = p.parse_args(argv)
    s, sched = prepare(args)
    ph, record = setups.photos(args, s, args.split, args.n, args.skip)
    (args.out_dir / "photos.json").write_text(json.dumps(record, indent=2))
    cfg = {k: str(v) for k, v in vars(args).items()} | {"t_noise": sched.noise_t, "t_first_step": sched.grid[0],
                                                       "steps": len(sched.grid) - 1, "scales": dime.SCALES}
    (args.out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (args.out_dir / "images").mkdir()
    for i, name in enumerate(ph.names):
        save_png(ph.x[i], args.out_dir / "images" / f"{Path(name).stem}_orig.png")
    rows, per_seed = counterfactuals(args, s, ph, args.peek, args.seeds, sched, args.out_dir)
    save_grid([torch.stack([c[i] for c in (ph.x, *per_seed)]) for i in range(len(ph.x))], args.out_dir / "grid.jpg")
    print(f"{len(rows)} explanations, {np.mean([r['flipped'] for r in rows]):.0%} flipped, "
          f"{np.mean([r['generator_calls'] for r in rows]):.0f} generator calls and "
          f"{np.mean([r['seconds'] for r in rows]):.2f} s each -> {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
