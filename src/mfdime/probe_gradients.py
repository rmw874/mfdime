"""Probe H3's classifier gradients along a fixed shortcut-guided DiME edit.

    PYTHONPATH=src uv run python -P -m mfdime.probe_gradients --grid-indices 1 30 59 --checkpoint-walk

Use --states direct for the directly noised control. Grid index 0 is only
available in that control because DiME does not use a look-ahead at index 0.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from mfdime import config, dime, flow, setups
from mfdime.common import key_seed


@dataclass(frozen=True)
class Gradients:
    shortcut_jump: Tensor
    exact_jump: Tensor
    exact_walk: Tensor
    shortcut_walk: Tensor


@torch.enable_grad()
def guidance_gradients(
    z: Tensor,
    grid: Sequence[float],
    v: flow.Velocity,
    u: flow.MeanVelocity,
    loss: Callable[[Tensor], Tensor],
    *,
    checkpoint_walk: bool = False,
) -> Gradients:
    """Return four classifier-loss gradients for the same noisy state.

    The shortcut and exact jump share the same clean endpoint and loss
    derivative. The exact gradients differentiate to the noisy flow state.
    """
    t = grid[0]
    z = z.detach().requires_grad_(True)
    jump = flow.jump(u, z, t, 0.0)
    at_endpoint = torch.autograd.grad(loss(jump), jump)[0]
    shortcut_scale = dime.vp_scale(t) / (1 - t)
    shortcut = at_endpoint * shortcut_scale
    exact_jump = torch.autograd.grad(jump, z, grad_outputs=at_endpoint)[0]

    def checkpointed_velocity(x: Tensor, time: float) -> Tensor:
        return checkpoint(v, x, time, use_reentrant=False)

    walk = flow.euler(checkpointed_velocity if checkpoint_walk else v, z, list(grid))
    at_walk_endpoint = torch.autograd.grad(loss(walk), walk)[0]
    exact_walk = torch.autograd.grad(walk, z, grad_outputs=at_walk_endpoint)[0]
    return Gradients(shortcut, exact_jump, exact_walk, at_walk_endpoint * shortcut_scale)


def trajectory_states(
    setup: setups.Setup,
    image: Tensor,
    clean: Tensor,
    target: Tensor,
    eps: Tensor,
    schedule: dime.Schedule,
    indices: set[int],
    noise_key: str,
    scale: float,
) -> dict[int, Tensor]:
    """Capture pre-update states from one jump1 shortcut pass at a fixed scale."""
    v = setup.twins.v(setup.null_label)
    u = setup.twins.u(setup.null_label)
    jump = dime.jump_peek(u, schedule.grid, 1)
    states = {}

    def capture(z: Tensor, k: int, v_z: Tensor) -> Tensor:
        if k in indices:
            states[k] = z.detach().to("cpu", copy=True)
        return jump.fn(z, k, v_z)

    generator = torch.Generator().manual_seed(key_seed(f"{noise_key}:scale0"))
    dime.guided_run(
        v, dime.Peek(capture, jump.calls), setup.ae.decode, setup.guide, setup.perc,
        image, clean, dime.start_latent(clean, eps, schedule), target, scale,
        schedule.grid, [generator], setup.use_logits,
    )
    return states


def summarize(grads: Gradients, cutoff: float) -> dict:
    """Compute direction and spatial power metrics for a single probe."""
    tensors = {
        "shortcut_jump": grads.shortcut_jump.float(),
        "exact_jump": grads.exact_jump.float(),
        "exact_walk": grads.exact_walk.float(),
        "shortcut_walk": grads.shortcut_walk.float(),
    }
    norms = {name: grad.norm() for name, grad in tensors.items()}

    def cosine(a: str, b: str) -> float | None:
        if min(norms[a].item(), norms[b].item()) <= 1e-12:
            return None
        dot = torch.dot(tensors[a].flatten(), tensors[b].flatten())
        return (dot / (norms[a] * norms[b])).clamp(-1, 1).item()

    height, width = grads.exact_jump.shape[-2:]
    device = grads.exact_jump.device
    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.fftfreq(width, device=device)
    high = fy[:, None].square() + fx[None, :].square() >= cutoff ** 2
    high_power, high_fraction = {}, {}
    for name, grad in tensors.items():
        power = torch.fft.fft2(grad, norm="ortho").abs().square()
        high_power[name] = power[..., high].sum().item()
        total = power.sum().item()
        high_fraction[name] = high_power[name] / total if total > 1e-24 else None

    return {
        "norms": {name: norm.item() for name, norm in norms.items()},
        "cosine_shortcut_exact_jump": cosine("shortcut_jump", "exact_jump"),
        "cosine_shortcut_exact_walk": cosine("shortcut_walk", "exact_walk"),
        "cosine_exact_jump_walk": cosine("exact_jump", "exact_walk"),
        "high_frequency_power": high_power,
        "high_frequency_fraction": high_fraction,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    setups.add_args(parser)
    parser.set_defaults(dataset="celebahq")
    parser.add_argument("--split", choices=("practice", "held_out"), default="practice")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, action="append", help="repeat for multiple seeds (default 0)")
    parser.add_argument("--states", choices=("trajectory", "direct"), default="trajectory")
    parser.add_argument("--gain", type=float, help="trajectory gain at the first DiME scale (default: dataset gain)")
    parser.add_argument("--frequency-cutoff", type=float, default=0.25, help="radial cutoff in cycles per spatial sample")
    parser.add_argument("--grid-indices", type=int, nargs="+", required=True)
    parser.add_argument("--checkpoint-walk", action="store_true")
    args = parser.parse_args(argv)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.smoke = False
    schedule = dime.dime_schedule()
    grid = schedule.grid
    first = 1 if args.states == "trajectory" else 0
    if any(not first <= k < len(grid) - 1 for k in args.grid_indices):
        parser.error(f"--grid-indices for {args.states} states must be between {first} and {len(grid) - 2}")
    if not 0 < args.frequency_cutoff <= 0.5:
        parser.error("--frequency-cutoff must be in (0, 0.5]")
    gain = args.gain if args.gain is not None else config.GAIN[args.dataset]
    scale = dime.SCALES[0] * gain if args.states == "trajectory" else None
    if args.n < 1 or args.skip < 0:
        parser.error("--n must be positive and --skip must be nonnegative")

    setup = setups.build(args)
    photos, _ = setups.photos(args, setup, args.split, args.n, args.skip)
    v = setup.twins.v(setup.null_label)
    u = setup.twins.u(setup.null_label)
    for name, image, label in zip(photos.names, photos.x, photos.target, strict=True):
        target = label.reshape(1).to(args.device)
        image = image.unsqueeze(0).to(args.device)
        with torch.no_grad():
            clean = setup.ae.encode(image)

        def classifier_loss(endpoint: Tensor, target: Tensor = target) -> Tensor:
            img = dime.clamp_through(setup.ae.decode(endpoint))
            return dime.class_loss(setup.guide(img), target, setup.use_logits)

        for seed in dict.fromkeys(args.seed or [0]):
            noise_key = f"{name}:seed{seed}"
            generator = torch.Generator().manual_seed(key_seed(f"{noise_key}:start"))
            eps = torch.randn(clean.shape, generator=generator, dtype=clean.dtype).to(clean.device)
            states = (
                trajectory_states(setup, image, clean, target, eps, schedule, set(args.grid_indices), noise_key, scale)
                if scale is not None else {}
            )

            for k in sorted(set(args.grid_indices)):
                t = grid[k]
                z = states[k].to(args.device) if scale is not None else flow.noise_to(clean, t, eps)
                cuda = z.is_cuda
                if cuda:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                grads = guidance_gradients(z, grid[k:], v, u, classifier_loss, checkpoint_walk=args.checkpoint_walk)
                if cuda:
                    torch.cuda.synchronize()
                row = {
                    "dataset": args.dataset, "photo": name, "seed": seed, "target": int(label),
                    "states": args.states, "trajectory_scale": scale, "k": k, "t": t,
                    "walk_steps": len(grid) - k - 1, "checkpoint_walk": args.checkpoint_walk,
                    "frequency_cutoff": args.frequency_cutoff,
                    "probe_seconds": time.perf_counter() - start,
                    "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else None,
                }
                print(json.dumps(row | summarize(grads, args.frequency_cutoff), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
