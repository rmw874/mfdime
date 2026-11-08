"""Summarize one H3 protocol from probe JSONL files, with photo-bootstrap intervals.

    PYTHONPATH=src uv run python -P -m mfdime.score_gradients probes.jsonl --out-dir results/h3_score
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

PROTOCOL = ("dataset", "states", "trajectory_scale", "frequency_cutoff")
SHORTCUT_COSINES = ("cosine_shortcut_exact_jump", "cosine_shortcut_exact_walk")
EXACT_COSINE = "cosine_exact_jump_walk"
FREQUENCY = ("high_frequency_fraction", "high_frequency_power")
COLORS = ("#1766a5", "#ac4c22")


def read_probes(paths: list[Path]) -> tuple[dict, dict]:
    rows = {}
    protocol = None
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                config = {key: row[key] for key in PROTOCOL}
                key = (row["photo"], row["seed"], row["k"])
                if "cosine_shortcut_exact_walk" not in row:
                    raise ValueError("walk-shortcut measurements are missing. Regenerate with the updated probe")
                if protocol is not None and config != protocol:
                    raise ValueError("mixed protocols. Score each state source and configuration separately")
                if key in rows:
                    raise ValueError(f"duplicate photo/seed/step {key}")
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            protocol = config
            rows[key] = row
    if not rows:
        raise ValueError("no probe rows found")
    return protocol, rows


def difference(row: dict, metric: str, method: str) -> float | None:
    shortcut = row[metric][f"shortcut_{method}"]
    exact = row[metric][f"exact_{method}"]
    return shortcut - exact if shortcut is not None and exact is not None else None


def interval(point: float, samples: np.ndarray, n: int) -> dict:
    ci = np.quantile(samples, [0.025, 0.975]).tolist() if n > 1 else None
    return {"mean": float(point), "ci95": ci}


def score(rows: dict, bootstrap: int, seed: int, alignment: float) -> dict:
    photos, seeds, steps = map(sorted, map(set, zip(*rows, strict=True)))
    if len(rows) != len(photos) * len(seeds) * len(steps):
        raise ValueError("incomplete photo × seed × step grid")
    if any(len({rows[p, s, k]["t"] for p in photos for s in seeds}) != 1 for k in steps):
        raise ValueError("inconsistent noise levels for the same step")
    noise = [rows[photos[0], seeds[0], k]["t"] for k in steps]
    if not np.all(np.isfinite(noise)) or any(a <= b for a, b in zip(noise[:-1], noise[1:], strict=True)):
        raise ValueError("noise levels must be finite and strictly decreasing over the edit")
    readers: dict[str, Callable[[dict], float | None]] = {
        metric: lambda row, metric=metric: row[metric] for metric in (*SHORTCUT_COSINES, EXACT_COSINE)
    }
    for metric in FREQUENCY:
        for method in ("jump", "walk"):
            readers[f"{metric}_{method}_difference"] = lambda row, metric=metric, method=method: difference(row, metric, method)

    rng = np.random.default_rng(seed)
    metrics = {}
    for name, read in readers.items():
        values = np.array([[[read(rows[p, s, k]) for k in steps] for s in seeds] for p in photos], dtype=float)
        # Keep a fixed photo cohort within each metric so changes across steps are paired.
        values = values[np.isfinite(values).all(axis=(1, 2))].mean(axis=1)
        n = len(values)
        if not n:
            metrics[name] = {"photos": 0, "excluded_photos": len(photos), "by_step": [], "late_minus_early": None}
            continue
        resampled = np.stack([values[rng.integers(n, size=n)].mean(axis=0) for _ in range(bootstrap)])
        mean = values.mean(axis=0)
        metrics[name] = {
            "photos": n,
            "excluded_photos": len(photos) - n,
            "by_step": [interval(mean[i], resampled[:, i], n) for i in range(len(steps))],
            "late_minus_early": interval(mean[-1] - mean[0], resampled[:, -1] - resampled[:, 0], n) if len(steps) > 1 else None,
        }

    return {
        "photos": len(photos), "seeds": seeds, "steps": steps, "noise": noise,
        "bootstrap": bootstrap, "bootstrap_seed": seed, "alignment_benchmark": alignment,
        "primary_frequency_measure": "fraction", "metrics": metrics,
    }


def plot(result: dict) -> str:
    """Render dependency-free SVG. Points are means and vertical bars are 95% intervals."""
    panels = [
        ("Shortcut vs. exact direction", SHORTCUT_COSINES, ("Jump", "Walk"), 0),
        ("Exact jump vs. exact walk direction", (EXACT_COSINE,), ("Exact",), result["alignment_benchmark"]),
        ("High-frequency power fraction (primary)", tuple(f"{FREQUENCY[0]}_{m}_difference" for m in ("jump", "walk")), ("Jump", "Walk"), 0),
        ("Absolute high-frequency power", tuple(f"{FREQUENCY[1]}_{m}_difference" for m in ("jump", "walk")), ("Jump", "Walk"), 0),
    ]
    steps = result["steps"]
    left, width, height = 82, 660, 125
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="780" height="1020" viewBox="0 0 780 1020">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="sans-serif" font-size="12" fill="#222">']
    for index, (title, keys, labels, baseline) in enumerate(panels):
        top = 52 + index * 245
        subtitle = "Cosine similarity" if index < 2 else "Shortcut minus exact"
        if index == 1:
            subtitle += f" | dashed line = {baseline:.2f} descriptive benchmark"
        svg.extend([f'<text x="{left}" y="{top - 29}" font-size="16" font-weight="bold">{title}</text>',
                    f'<text x="{left}" y="{top - 11}">{subtitle}</text>'])
        series = [result["metrics"][key]["by_step"] for key in keys]
        bounds = [baseline, *(v for points in series for p in points for v in (p["ci95"] or [p["mean"]]))]
        lo, hi = (-1, 1) if index < 2 else (min(bounds), max(bounds))
        if lo == hi:
            lo, hi = lo - 0.01, hi + 0.01

        def y(value: float, top: int = top, lo: float = lo, hi: float = hi) -> float:
            return top + (hi - value) / (hi - lo) * height

        xs = [left + width * ((k - steps[0]) / (steps[-1] - steps[0]) if len(steps) > 1 else 0.5) for k in steps]
        for value in np.linspace(lo, hi, 5):
            svg.append(f'<path d="M{left} {y(value):.2f}h{width}" stroke="#ddd"/>')
            svg.append(f'<text x="{left - 7}" y="{y(value) + 4:.2f}" text-anchor="end">{value:.3g}</text>')
        svg.append(f'<path d="M{left} {y(baseline):.2f}h{width}" stroke="#777" stroke-dasharray="5 4"/>')
        tick_indices = sorted(set(np.linspace(0, len(steps) - 1, min(7, len(steps)), dtype=int)))
        for i in tick_indices:
            svg.append(f'<text x="{xs[i]:.2f}" y="{top + height + 17}" text-anchor="middle">{steps[i]}</text>')
        svg.append(f'<text x="{left + width / 2}" y="{top + height + 36}" text-anchor="middle">DiME step</text>')
        for j, (points, label, color) in enumerate(zip(series, labels, COLORS, strict=False)):
            n = result["metrics"][keys[j]]["photos"]
            svg.append(f'<text x="{left + 200 * j}" y="{top + height + 55}" fill="{color}">{label} (photos: {n})</text>')
            if points:
                coords = " ".join(f'{x:.2f},{y(p["mean"]):.2f}' for x, p in zip(xs, points, strict=True))
                svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
            for x, point in zip(xs, points, strict=bool(points)):
                if point["ci95"] is not None:
                    a, b = point["ci95"]
                    svg.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{y(a):.2f}" y2="{y(b):.2f}" stroke="{color}"/>')
                svg.append(f'<circle cx="{x:.2f}" cy="{y(point["mean"]):.2f}" r="3" fill="{color}"/>')
    return "\n".join([*svg, "</g>", "</svg>"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alignment-benchmark", type=float, default=0.90)
    args = parser.parse_args(argv)
    if args.bootstrap < 1 or args.seed < 0 or not -1 <= args.alignment_benchmark <= 1:
        parser.error("use a positive bootstrap count, nonnegative seed, and alignment benchmark in [-1, 1]")
    try:
        protocol, rows = read_probes(args.probes)
        result = score(rows, args.bootstrap, args.seed, args.alignment_benchmark) | {"protocol": protocol}
        figure = plot(result)
        args.out_dir.mkdir(parents=True)
        (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        (args.out_dir / "curves.svg").write_text(figure)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"STOP: {exc}\n")


if __name__ == "__main__":
    main()
