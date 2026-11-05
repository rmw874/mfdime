"""Score the completed held-out H2 experiment, retaining numerical failures in all success-rate denominators.

PYTHONPATH=src python -P -m mfdime.h2_score --runs results/h2_full_v1 --out-dir results/h2_full_v1/score
Paired bootstrap: resample photos within class pair; keep all seeds and arms together. Pooled results weight photos
equally. Report jump1 and jump4 separately, with no automatic hypothesis-support label or pilot data mixed in.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from mfdime import h2


def load_results(root: Path) -> tuple[dict, dict, dict]:
    wrapper = h2.read_json(root / "protocol.json")
    protocol, signature = wrapper["protocol"], wrapper["signature"]
    if protocol.get("schema") != 1 or h2.digest(protocol) != signature or protocol.get("split") != "held_out":
        raise ValueError("invalid H2 protocol or split")
    for group, seeds in enumerate(protocol["seed_groups"]):
        marker = root / "jobs" / f"group{group}" / "complete.json"
        if not marker.exists():
            raise ValueError(f"incomplete seed group {group}: {marker}")
        complete = h2.read_json(marker)
        if complete["signature"] != signature or complete["seeds"] != seeds:
            raise ValueError("seed group does not match protocol")
    cohorts, results = {}, {}
    for source, target in protocol["pairs"]:
        pair = f"{source}_{target}"
        record = h2.read_json(root / "cohorts" / f"{pair}.json")
        h2.validate_cohort(record, signature, source, target)
        cohorts[pair] = record
        results[pair] = {}
        for peek, mode in protocol["arms"]:
            arm = f"{peek}/{mode}"
            results[pair][arm] = []
            for seed in protocol["seeds"]:
                folder = root / "runs" / pair / f"seed{seed}" / peek / mode
                marker = folder / "complete.json"
                if not marker.exists():
                    raise ValueError(f"incomplete arm: {folder}")
                complete = h2.read_json(marker)
                if complete["signature"] != signature or complete["cohort_sha256"] != record["cohort_sha256"]:
                    raise ValueError(f"arm identity mismatch: {folder}")
                rows = [json.loads(line) for line in (folder / "manifest.jsonl").read_text().splitlines()]
                if complete["rows_sha256"] != h2.digest(rows) or len(rows) != len(record["names"]):
                    raise ValueError(f"damaged or incomplete arm manifest: {folder}")
                if [r["photo"] for r in rows] != record["names"]:
                    raise ValueError(f"arm does not match frozen photo ordering: {folder}")
                for row in rows:
                    identity = {"signature": signature, "cohort_sha256": record["cohort_sha256"], "source": source,
                                "target": target, "photo": row["photo"], "seed": seed, "peek": peek, "gradient_mode": mode}
                    if any(row.get(k) != identity[k] for k in ("source", "target", "photo", "seed", "peek", "gradient_mode")):
                        raise ValueError("sample labels do not match the expected arm")
                    if h2.completed_sample(h2.sample_dir(folder, row["photo"]), identity) != row:
                        raise ValueError("arm and sample checkpoint disagree")
                    if row["status"] == "numerical_failure":
                        if row["flipped"] or row["check_flipped"]:
                            raise ValueError("numerical failure counted as a success")
                    elif (row["flipped"] != (row["guide_after"] == target)
                          or row["check_flipped"] != (row["check_after"] == target)):
                        raise ValueError("prediction and success flag disagree")
                results[pair][arm].append(rows)
    if not any(c["names"] for c in cohorts.values()):
        raise ValueError("no eligible photos")
    return protocol, cohorts, results


def arm_summary(rows: list[dict], seeds: list[int]) -> dict:
    n = len(rows)
    if not n:
        return {"explanations": 0, "distinct_photos": 0, "status": "no_eligible_photos"}
    guide = sum(r["flipped"] for r in rows)
    check = sum(r["check_flipped"] for r in rows)
    joint = sum(r["flipped"] and r["check_flipped"] for r in rows)
    summary = {"explanations": n, "distinct_photos": len({r["photo"] for r in rows}),
               "guide_success": guide / n, "check_success": check / n, "joint_success": joint / n,
               "check_success_among_guide_successes": joint / guide if guide else None,
               "numerical_failures": sum(r["status"] == "numerical_failure" for r in rows),
               "check_success_per_seed": [sum(r["check_flipped"] for r in rows if r["seed"] == seed)
                                          / sum(r["seed"] == seed for r in rows) for seed in seeds],
               "scale_counts": {str(scale): sum(r["scale"] == scale for r in rows) for scale in (8.0, 10.0, 15.0, None)}}
    for metric in ("pixel_l1", "perceptual", "seconds", "generator_calls", "batch_peak_allocated_bytes",
                   "batch_peak_reserved_bytes"):
        values = [float(r[metric]) for r in rows if r.get(metric) is not None and math.isfinite(r[metric])]
        summary[metric] = {"mean": sum(values) / len(values) if values else None,
                           "max": max(values) if values else None, "observations": len(values)}
    return summary


def analyse(protocol: dict, cohorts: dict, results: dict, boot: int, seed: int) -> dict:
    """Bootstrap per-photo seed-averaged differences, stratified by source/target pair."""
    if boot < 1:
        raise ValueError("bootstrap draws must be positive")
    arms = [f"{peek}/{mode}" for peek, mode in protocol["arms"]]
    peeks = [peek for peek, mode in protocol["arms"] if mode == "exact"]
    summary: dict = {"scope": "held_out", "photos": sum(len(c["names"]) for c in cohorts.values()),
                     "seeds": protocol["seeds"], "bootstrap_draws": boot, "bootstrap_seed": seed,
                     "weighting": "equal weight per photo; seeds averaged; photos resampled within class pair",
                     "arms": {}, "pairs": {}, "comparisons": {}}
    pair_deltas = {}
    for pair, pair_results in results.items():
        n = len(cohorts[pair]["names"])
        pair_summary = {"photos": n, "arms": {}, "comparisons": {}}
        for arm in arms:
            rows = [r for group in pair_results[arm] for r in group]
            pair_summary["arms"][arm] = arm_summary(rows, protocol["seeds"])
        summary["pairs"][pair] = pair_summary
        if n:
            pair_deltas[pair] = {}
            for peek in peeks:
                a = np.array([[r["check_flipped"] for r in group] for group in pair_results[f"{peek}/shortcut"]], dtype=float)
                b = np.array([[r["check_flipped"] for r in group] for group in pair_results[f"{peek}/exact"]], dtype=float)
                pair_deltas[pair][peek] = (b - a).mean(axis=0)
    for arm in arms:
        all_rows = [r for pair in results.values() for group in pair[arm] for r in group]
        summary["arms"][arm] = arm_summary(all_rows, protocol["seeds"])
    rng = np.random.default_rng(seed)
    pooled = {peek: np.zeros(boot) for peek in peeks}
    for pair, values in pair_deltas.items():
        n = len(cohorts[pair]["names"])
        # The same photo draws are shared across jump counts; all seeds remain together in the per-photo average.
        idx = rng.integers(0, n, size=(boot, n))
        for peek in peeks:
            draws = values[peek][idx].mean(axis=1)
            lo, hi = np.percentile(draws, [2.5, 97.5]).tolist()
            summary["pairs"][pair]["comparisons"][peek] = {
                "difference_exact_minus_shortcut": float(values[peek].mean()), "ci95": [lo, hi]}
            pooled[peek] += draws * n / summary["photos"]
    for peek in peeks:
        a, b = summary["arms"][f"{peek}/shortcut"], summary["arms"][f"{peek}/exact"]
        lo, hi = np.percentile(pooled[peek], [2.5, 97.5]).tolist()
        summary["comparisons"][peek] = {"difference_exact_minus_shortcut": b["check_success"] - a["check_success"],
                                         "ci95": [lo, hi]}
    return summary


def markdown(result: dict) -> str:
    lines = ["# Held-out H2 experiment", "",
             f"{result['photos']} distinct photos × {len(result['seeds'])} seeds. All four arms use the same frozen cohort.",
             "", "Primary outcome: independent ViT target success over all eligible originals, including guide failures.",
             "Numerical failures count as unsuccessful outcomes; finite corrupted images are retained.", "",
             "| Arm | Guide success | Evaluator success | Joint success | Numerical failures |",
             "|---|---|---|---|---|"]
    for name, arm in result["arms"].items():
        lines.append(f"| {name} | {arm['guide_success']:.1%} | {arm['check_success']:.1%} | "
                     f"{arm['joint_success']:.1%} | {arm['numerical_failures']} |")
    lines += ["", "| Exact minus shortcut | Difference (pp) | Paired 95% interval (pp) |", "|---|---|---|"]
    for peek, comparison in result["comparisons"].items():
        lo, hi = comparison["ci95"]
        lines.append(f"| {peek} | {100 * comparison['difference_exact_minus_shortcut']:+.2f} | [{100 * lo:+.2f}, {100 * hi:+.2f}] |")
    lines += ["", "Photos are resampled within class pair; all five seeds and all arms stay paired. Pooled results weight photos equally.",
              "Intervals describe each prespecified comparison separately; there is no automatic overall H2-support label.",
              "Independent-classifier transfer is a semantic-validity proxy, not proof of visual realism.", "",
              "| Pair | Photos | jump1 shortcut / exact | jump4 shortcut / exact |", "|---|---|---|---|"]
    for name, pair in result["pairs"].items():
        if not pair["photos"]:
            lines.append(f"| {name} | 0 | No eligible photos | No eligible photos |")
        else:
            values = [f"{pair['arms'][f'{peek}/shortcut']['check_success']:.1%} / "
                      f"{pair['arms'][f'{peek}/exact']['check_success']:.1%}" for peek in ("jump1", "jump4")]
            lines.append(f"| {name} | {pair['photos']} | {values[0]} | {values[1]} |")
    lines += ["", "Secondary image-change, perceptual, timing, memory, scale and per-seed metrics are in score.json.",
              "Continuous metrics report their available-observation counts; numerical failures have no final-image distance.",
              "Timings include guidance, final decoding and diagnostics, but exclude evaluator scoring and setup.",
              "The experiment retains the pilot's execution settings; exact replay across jobs is not guaranteed."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.out_dir.exists():
        p.error("score output already exists; choose a new --out-dir")
    if args.boot < 1:
        p.error("--boot must be positive")
    protocol, cohorts, results = load_results(args.runs)
    result = analyse(protocol, cohorts, results, args.boot, args.seed)
    result["experiment_signature"] = h2.digest(protocol)
    args.out_dir.mkdir(parents=True)
    h2.write_json(args.out_dir / "score.json", result)
    text = markdown(result)
    (args.out_dir / "score.md").write_text(text)
    print(text)


if __name__ == "__main__":
    sys.exit(main())
