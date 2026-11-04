"""DiME's guided loop as published (Jeanneret et al., ACCV 2022; code github.com/guillaumejs2403/DiME at 3267bbf, MIT),
transcribed for a flow model on the linear path z_t = (1 - t) x0 + t eps. Only the peek is swappable.

What is DiME's, and where it is in their code:
  - noise levels: a 500-step linear DDPM respaced to 200 steps (README / test.sh: S=60, --diffusion_steps 500
    --noise_schedule linear --timestep_respacing 200), mapped to flow time by matching the signal-to-noise ratio. The
    input is noised to level 60 (t = 0.5548; main.py: t = start_step, q_sample) and the model then runs at levels 59
    down to 0 (t = 0.5498 ... 0.0139; sample_utils.py: indices = range(start_step)[::-1]), so the first step reads
    level-60 noise as level 59, as DiME's code does: 60 guided steps          [dime_schedule, start_latent]
  - each step: ancestral (stochastic) sample of the posterior q(z_s | z_t, x0_hat), mean shifted by -variance * grad,
    no noise on the last step (sample_utils.py, get_DiME_iterative_sampling: out["mean"] - out["variance"] * grads);
    the last step's variance is the step before's, not 0 (gaussian_diffusion.py: posterior_log_variance_clipped), so
    the last step still takes the nudge; x0_hat clipped to [-1, 1] here and in the peek's walk (p_mean_variance,
    clip_denoised)
  - the nudge is computed on x_hat: the original image at the first step, then the peek of the current noisy image
    (x_t = img.clone(); x_t recomputed by the inner p_sample_loop after every step)
  - grad = s * dCE/dx_hat / alpha + l1 * d||z - x||_1/dz + dPerc/dx_hat / alpha, alpha = sqrt(alphabar)
    (clean_class_cond_fn / clean_multiclass_cond_fn, dist_cond_fn); l1 = 0.05, perceptual = 30 x MSE of VGG19
    features[:18] on clamped, ImageNet-normalised images (test.sh: L1=0.05, PERC=30, LAYER=18; PerceptualLoss,
    Normalizer); the classifier gets the peek as it is (main.py: ClassificationModel, no Normalizer)
  - classifier scales 8, 10, 15: every image not yet flipped is redone from the same noisy start at the next scale
    (main.py: classifier_scales='8,10,15', the `transformed` loop)
  - DiME's own peek walks every remaining step with stochastic sampling (use_sampling_on_x_t True): 1830 calls

DiME's step in flow coordinates, exactly: at the same signal-to-noise ratio the flow's noisy image is c(t) times the
DDPM's (vp_scale), so DiME's update reads, for ours, grad = (s dCE/dx_hat + dPerc/dx_hat) c(t) / (1 - t) / c(s)
+ l1 sign(z / c(t) - x) / c(s), with the posterior's mean and variance in flow coordinates.

What differs, forced by the setting: the generator is a flow model; on ImageNet it is a latent one (SiT/DMF), so z,
x_hat and the L1 term live in SD-VAE latent space and the losses see decoded images; on faces pMF works in pixels, as
DiME does, its wrappers in mfdime.pmf clip x0_hat, and the losses see pMF's auxiliary head's output (pmf.PixelSpace,
as FiRe's samplers use pMF); latents have no fixed range and are not clipped. The step variance is the posterior's
(DiME's model learns it, between the clipped posterior variance and beta). The decoded peek is clamped to [-1, 1]
before the classifier, the range DiME's clipped peek is in, with the gradient passed through (clamp_through), as DiME's
graph has no clamp there. The multi-class loss is log-softmax (DiME's published runs
used a binary classifier). DiME's scales were tuned on CelebA pixels, so one gain multiplies them (as ACE did for
CelebA-HQ), the same for every peek. DiME runs in fp16, ours in fp32.

Only the peek differs between arms (peek): walk_rest (DiME's own: the remaining steps with fresh noise at each),
euler_rest (the same remaining steps without noise: the flow's own path, the reference) and jump<N> (N flow-map jumps
along that path). DiME's guided steps keep their noise in every arm, drawn per image, so arms pair image by image
whatever the batch.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mfdime import flow
from mfdime.common import key_seed

if TYPE_CHECKING:
    from torchvision.models import VGG

SCALES = (8.0, 10.0, 15.0)
L1, PERC, PERC_LAYER = 0.05, 30.0, 18


def respaced_times(respacing: int = 200, diffusion_steps: int = 500) -> np.ndarray:
    """The flow times of the respaced DDPM's levels, least noisy first (SNR match, (1 - t)^2 / t^2 = alphabar /
    (1 - alphabar))."""
    scale = 1000 / diffusion_steps  # guided-diffusion's linear schedule
    alphabar = np.cumprod(1 - np.linspace(scale * 1e-4, scale * 0.02, diffusion_steps, dtype=np.float64))
    frac, cur, use = (diffusion_steps - 1) / (respacing - 1), 0.0, []
    for _ in range(respacing):  # respace.space_timesteps with one section
        use.append(round(cur))
        cur += frac
    ab = alphabar[sorted(set(use))]
    return 1 / (1 + np.sqrt(ab / (1 - ab)))


@dataclass(frozen=True)
class Schedule:
    """DiME's noise levels as flow times: noise_t, the level the input is noised to; grid, the levels the model runs at
    (noisiest first), then 0, the clean image (start_step + 1 points, start_step steps)."""

    noise_t: float
    grid: list[float]


def dime_schedule(start_step: int = 60, respacing: int = 200, diffusion_steps: int = 500) -> Schedule:
    """As DiME's code runs: the input noised to level start_step, the model then run at levels start_step - 1 down
    to 0."""
    t = respaced_times(respacing, diffusion_steps)
    return Schedule(float(t[start_step]), [float(x) for x in t[start_step - 1::-1]] + [0.0])


def clamp_through(x: Tensor) -> Tensor:
    """x clamped to [-1, 1] in value, with the gradient passed through unchanged (a straight-through clamp)."""
    return x + (x.clamp(-1, 1) - x).detach()


def vp_scale(t: float) -> float:
    """c(t) = sqrt((1 - t)^2 + t^2): the flow's noisy image (1 - t) x + t eps is c(t) times the DDPM's
    sqrt(alphabar) x + sqrt(1 - alphabar) eps at the same signal-to-noise ratio."""
    return math.sqrt((1 - t) ** 2 + t ** 2)


def start_latent(x: Tensor, eps: Tensor, sched: Schedule) -> Tensor:
    """DiME's starting noisy image in flow coordinates: the DDPM's noisy image at level noise_t, read at the first
    model level grid[0], so c(grid[0]) times the DDPM's."""
    return (vp_scale(sched.grid[0]) / vp_scale(sched.noise_t)) * flow.noise_to(x, sched.noise_t, eps)


def class_loss(logits: Tensor, target: Tensor, use_logits: bool) -> Tensor:
    """DiME's classifier term, summed over the batch (clean_class_cond_fn / clean_multiclass_cond_fn): the target's
    log-probability, or with use_logits its logit (for a two-way classifier, minus the other logit: DiME's binary
    -(y l - (1 - y) l)). Binary classifiers enter as two-column logits [0, l]."""
    if not use_logits:
        return F.cross_entropy(logits, target, reduction="sum")
    picked = logits.gather(1, target[:, None])[:, 0]
    if logits.shape[1] == 2:
        picked = picked - logits.gather(1, (1 - target)[:, None])[:, 0]
    return -picked.sum()


def posterior(z: Tensor, x0: Tensor, t: float, s: float) -> tuple[Tensor, float]:
    """Mean and variance of q(z_s | z_t, x0) for the Gaussian path alpha = 1 - t, sigma = t (s < t)."""
    a_ts = (1 - t) / (1 - s)
    var_ts = t ** 2 - a_ts ** 2 * s ** 2
    mean = (a_ts * s ** 2 / t ** 2) * z + ((1 - s) * var_ts / t ** 2) * x0
    return mean, var_ts * s ** 2 / t ** 2


class Perceptual(nn.Module):
    """DiME's PerceptualLoss: c x MSE of VGG19 features[:layer] on clamped, ImageNet-normalised images, summed over
    the batch."""

    mu: Tensor
    sigma: Tensor

    def __init__(self, vgg19: VGG, layer: int = PERC_LAYER, c: float = PERC):
        super().__init__()
        self.features = nn.Sequential(*list(vgg19.features.children())[:layer])
        self.c = c
        self.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("sigma", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def embed(self, x: Tensor) -> Tensor:
        return self.features(((x.clamp(-1, 1) + 1) / 2 - self.mu) / self.sigma).flatten(1)

    def forward(self, x0: Tensor, x1: Tensor) -> Tensor:
        return self.c * F.mse_loss(self.embed(x0), self.embed(x1), reduction="none").mean(1).sum()


@dataclass
class Peek:
    """x_hat (latent) of a noisy latent at grid index k, given the step's own velocity v_z, and the generator calls it
    costs beyond that one."""

    fn: Callable[[Tensor, int, Tensor], Tensor]
    calls: Callable[[int], int]


def walk_rest_peek(v: Callable[[Tensor, float], Tensor], grid: list[float], gen: torch.Generator) -> Peek:
    """DiME's peek: unguided ancestral sampling over every remaining step of the grid."""
    def fn(z: Tensor, k: int, v_z: Tensor) -> Tensor:  # DiME recomputes the first velocity too: 1830 calls in all
        for t, s in zip(grid[k:-1], grid[k + 1:], strict=True):
            mean, var = posterior(z, z - t * v(z, t), t, s)
            z = mean + (var ** 0.5) * torch.randn(z.shape, generator=gen).to(z.device) if s > 0 else mean
        return z
    return Peek(fn, lambda k: len(grid) - 1 - k)


def euler_rest_peek(v: flow.Velocity, grid: list[float]) -> Peek:
    """The reference: DiME's remaining steps walked without noise, the deterministic path a flow map's jumps estimate
    (the first step reuses the step's own velocity)."""
    return Peek(lambda z, k, v_z: flow.euler(v, z, grid[k:], v_first=v_z), lambda k: len(grid) - 2 - k)


def jump_peek(u: flow.MeanVelocity, grid: list[float], n: int) -> Peek:
    """Ours: n flow-map jumps from the current noise level to t = 0."""
    return Peek(lambda z, k, v_z: flow.jump(u, z, grid[k], 0.0, n), lambda k: n)


def peek(arm: str, v: flow.Velocity, u: flow.MeanVelocity, grid: list[float], gen: torch.Generator) -> Peek:
    """walk_rest, euler_rest or jump<N>."""
    if arm == "walk_rest":
        return walk_rest_peek(v, grid, gen)
    if arm == "euler_rest":
        return euler_rest_peek(v, grid)
    if m := re.fullmatch(r"jump(\d+)", arm):
        return jump_peek(u, grid, int(m.group(1)))
    raise SystemExit(f"STOP: unknown peek {arm!r} (walk_rest, euler_rest, jump<N>)")


def noise_like(z: Tensor, gens: list[torch.Generator]) -> Tensor:
    """Standard normal noise shaped like z, one generator per image."""
    return torch.stack([torch.randn(z.shape[1:], generator=g) for g in gens]).to(z.device)


def guided_run(v, peek: Peek, decode, clf, perc: Perceptual, x_img: Tensor, x_lat: Tensor, z_start: Tensor,
               target: Tensor, scale: float, grid: list[float], gens: list[torch.Generator],
               use_logits: bool = False) -> tuple[Tensor, int]:
    """One DiME pass at one classifier scale (target: one class per image; gens: each image's step-noise generator):
    the final latent and the generator calls spent per image."""
    z, x_hat, calls = z_start, x_lat, 0  # DiME's first nudge is taken on the original image (here: its latent)
    tgt = target.to(z.device)
    var_before = 0.0  # the step before's posterior variance, in the DDPM's coordinates
    for k, (t, s) in enumerate(zip(grid[:-1], grid[1:], strict=True)):
        with torch.no_grad():
            v_z = v(z, t)
            calls += 1
            if k > 0:
                x_hat = peek.fn(z, k, v_z)
                calls += peek.calls(k)
        xh = x_hat.detach().requires_grad_(True)
        img = decode(xh)
        loss = scale * class_loss(clf(clamp_through(img)), tgt, use_logits) + perc(img, x_img)
        g_hat = torch.autograd.grad(loss, xh)[0]
        c_t, c_s = vp_scale(t), vp_scale(s)
        grad = (g_hat * c_t / (1 - t) + L1 * torch.sign(z / c_t - x_lat)) / c_s  # L1: d/dz of l1 * ||z - x||_1
        with torch.no_grad():
            mean, var = posterior(z, z - t * v_z, t, s)
            mean = mean - (var if s > 0 else var_before * c_s ** 2) * grad
            z = mean + (var ** 0.5) * noise_like(z, gens) if s > 0 else mean
            var_before = var / c_s ** 2
    return z, calls


def explain(v, peek: Peek, decode, clf, perc: Perceptual, x_img: Tensor, x_lat: Tensor, z_start: Tensor,
            target: Tensor | int, grid: list[float], gain: float, noise_keys: list[str], scales=SCALES,
            use_logits: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    """DiME's scale schedule: images not flipped at one scale are redone from the same noisy start at the next.
    target: one class for all images or one per image. noise_keys: one string per image (its photo and seed) that fixes
    its step noise at every scale, the same in every arm and batch. Returns (final latents, scale index used per image
    or -1 if never flipped, generator calls per image)."""
    target = torch.as_tensor(target, dtype=torch.long).expand(len(z_start)).clone()
    out, used = z_start.clone(), torch.full((len(z_start),), -1, dtype=torch.long)
    calls = torch.zeros(len(z_start), dtype=torch.long)
    todo = torch.ones(len(z_start), dtype=torch.bool)
    for j, s in enumerate(scales):
        idx = todo.nonzero().flatten()
        dev = idx.to(z_start.device)
        gens = [torch.Generator().manual_seed(key_seed(f"{noise_keys[i]}:scale{j}")) for i in idx.tolist()]
        z, c = guided_run(v, peek, decode, clf, perc, x_img[dev], x_lat[dev], z_start[dev], target[idx], gain * s, grid,
                          gens, use_logits)
        out[dev], calls[idx] = z, calls[idx] + c
        with torch.no_grad():
            flipped = clf(decode(z).clamp(-1, 1)).argmax(1).cpu() == target[idx]
        used[idx[flipped]] = j
        todo[idx[flipped]] = False
        if not todo.any():
            break
    return out, used, calls
