"""The linear flow path x_t = (1 - t) x0 + t eps (noise at t = 1, data at t = 0) and the two ways to move along it.

The conventions are DMF's (third_party/dmf/loss.py, samplers.py), and pMF's are the same: a flow model predicts the
velocity v = eps - x0, so x0 = x_t - t v; a flow map predicts the average velocity u over [r, t], so x_r = x_t + (r - t) u.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import torch
from torch import Tensor

Velocity = Callable[[Tensor, float], Tensor]  # v(x_t, t)
MeanVelocity = Callable[[Tensor, float, float], Tensor]  # u(x_t, t, r)


def noise_to(x0: Tensor, t: float, eps: Tensor) -> Tensor:
    return (1 - t) * x0 + t * eps


def times(t_from: float, t_to: float, steps: int) -> list[float]:
    return torch.linspace(t_from, t_to, steps + 1, dtype=torch.float64).tolist()


def euler(v: Velocity, x: Tensor, ts: list[float], v_first: Tensor | None = None) -> Tensor:
    """Walk the flow ODE without noise through the times ts, one Euler step per gap (v_first: v(x, ts[0]), if already
    computed). This deterministic path is the one a flow map's jumps estimate."""
    for i, (a, b) in enumerate(pairwise(ts)):
        x = x + (b - a) * (v_first if i == 0 and v_first is not None else v(x, a))
    return x


def jump(u: MeanVelocity, x: Tensor, t_from: float, t_to: float, steps: int = 1) -> Tensor:
    """Flow-map jumps from t_from to t_to in uniform steps (one step is one call)."""
    ts = times(t_from, t_to, steps)
    for a, b in pairwise(ts):
        x = x + (b - a) * u(x, a, b)
    return x
