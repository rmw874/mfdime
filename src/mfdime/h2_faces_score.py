"""Score the complete held-out face experiment, paired over five seeds and stratified by smile-change direction.

  PYTHONPATH=src uv run --group faces python -P -m mfdime.h2_faces_score --runs results/h2_faces_full_v1 \
      --out-dir results/h2_faces_full_v1/score

Verifies the frozen cohort and complete paired coverage, and reports pooled results and adding and removing smiles
separately. Photos are bootstrapped within direction with all five seeds and all arms kept together, 5,000 draws, one
95% interval per exact-minus-shortcut comparison; pooled results weight faces equally. Secondary: guide and joint
success, evaluator success among guide successes, pixel and VGG distances, target-directed OpenCLIP cosine-margin
changes, scale usage, timing and peak memory. Cosine margins are not calibrated probabilities.

The evaluator (mfdime.clip_smile) is OpenCLIP's ViT-L-14 with the LAION-2B checkpoint laion2b_s32b_b82k at a pinned
revision. The predicted class has the larger image/text cosine similarity with one of two fixed prompts; the neutral
prompt stands in for the non-smiling class, which is why the pilot checks the evaluator on unfiltered originals first.
Originals and edits are converted to uint8 RGB, then given the checkpoint's evaluation transform (224 px, bicubic
resize and centre crop, mean and std 0.5).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from mfdime import h2, h2_faces, h2_score


def load_results(root: Path):
    wrapper = h2.read_json(root / "protocol.json")
    protocol, signature = wrapper["protocol"], wrapper["signature"]
    if (protocol.get("schema") != "faces_h2_v1" or h2.digest(protocol) != signature or protocol.get("batch") != 1
            or protocol.get("split") != "held_out" or protocol["seeds"] != list(h2_faces.SEEDS)
            or protocol["arms"] != [list(a) for a in h2_faces.ARMS]):
        raise ValueError("invalid full-faces protocol")
    cohort = h2.read_json(root / "cohort.json")
    h2_faces.validate_cohort(cohort, signature)
    if (len(cohort["names"]) != protocol["n"] or cohort["practice_count"] != protocol["practice_count"]
            or cohort["skip_tuning"] != protocol["skip_tuning"]):
        raise ValueError("cohort count does not match protocol")
    all_rows = {f"{peek}/{mode}": [] for peek, mode in h2_faces.ARMS}
    for seed in protocol["seeds"]:
        marker = root / "jobs" / f"seed{seed}" / "complete.json"
        if not marker.exists():
            raise ValueError(f"seed {seed} is incomplete")
        complete = h2.read_json(marker)
        if complete["signature"] != signature or complete["seed"] != seed or complete["photos"] != protocol["n"]:
            raise ValueError("seed completion identity mismatch")
        for peek, mode in h2_faces.ARMS:
            folder = root / "runs" / f"seed{seed}" / peek / mode
            complete = h2.read_json(folder / "complete.json")
            rows = [json.loads(line) for line in (folder / "manifest.jsonl").read_text().splitlines()]
            if (complete["signature"] != signature or complete["cohort_sha256"] != cohort["cohort_sha256"]
                    or complete["rows_sha256"] != h2.digest(rows) or [r["photo"] for r in rows] != cohort["names"]):
                raise ValueError(f"incomplete or mismatched arm: {folder}")
            for entry, row in zip(cohort["selected"], rows, strict=True):
                identity = h2_faces.sample_identity(signature, cohort, entry, seed, peek, mode)
                if any(row.get(k) != identity[k] for k in ("photo", "source", "target", "seed", "peek", "gradient_mode")):
                    raise ValueError("sample labels disagree with cohort or arm")
                if row.get("batch") != 1 or row.get("check_before") != entry["check_before"]:
                    raise ValueError("sample batch or original evaluator prediction mismatch")
                if h2.completed_sample(h2.sample_dir(folder, entry["photo"]), identity) != row:
                    raise ValueError("sample checkpoint and arm manifest disagree")
                if row["status"] == "numerical_failure":
                    if row["flipped"] or row["check_flipped"]:
                        raise ValueError("numerical failure counted as success")
                elif (row["flipped"] != (row["guide_after"] == row["target"])
                      or row["check_flipped"] != (row["check_after"] == row["target"])):
                    raise ValueError("prediction and success flags disagree")
            all_rows[f"{peek}/{mode}"].append(rows)
    cohorts, results = {}, {}
    for target, name in ((1, "add_smile"), (0, "remove_smile")):
        indices = [i for i, r in enumerate(cohort["selected"]) if r["target"] == target]
        cohorts[name] = {"names": [cohort["names"][i] for i in indices]}
        results[name] = {arm: [[rows[i] for i in indices] for rows in per_seed] for arm, per_seed in all_rows.items()}
    return protocol, cohorts, results


def analyse(protocol, cohorts, results, boot, seed):
    result = h2_score.analyse(protocol, cohorts, results, boot, seed)
    result["evaluator"] = {"repo": protocol["clip_repo"], "revision": protocol["clip_revision"], "prompts": protocol["prompts"]}
    result["weighting"] = "equal weight per photo; seeds averaged; bootstrap stratified by adding/removing smiles"
    for name, summary in [(None, result), *result["pairs"].items()]:
        for arm in result["arms"]:
            rows = [r for direction, group in results.items() if name is None or name == direction
                    for per_seed in group[arm] for r in per_seed]
            values = [r["target_margin_change"] for r in rows
                      if r.get("target_margin_change") is not None and math.isfinite(r["target_margin_change"])]
            summary["arms"][arm]["target_margin_change"] = {
                "mean": sum(values) / len(values) if values else None, "observations": len(values)}
    result["directions"] = result.pop("pairs")
    result["experiment_signature"] = h2.digest(protocol)
    return result


def markdown(result):
    lines = ["# Held-out faces H2", "", f"{result['photos']} distinct faces × {len(result['seeds'])} seeds, batch size one.",
             "Fixed STEEX guide and OpenCLIP evaluator; all arms share the same original-image cohort.", "",
             "| Arm | Guide success | OpenCLIP success | Joint success | Numerical failures |", "|---|---|---|---|---|"]
    for name, arm in result["arms"].items():
        lines.append(f"| {name} | {arm['guide_success']:.1%} | {arm['check_success']:.1%} | "
                     f"{arm['joint_success']:.1%} | {arm['numerical_failures']} |")
    lines += ["", "| Exact minus shortcut | OpenCLIP success difference (pp) | Paired 95% interval (pp) |", "|---|---|---|"]
    for peek, comparison in result["comparisons"].items():
        lo, hi = comparison["ci95"]
        lines.append(f"| {peek} | {100 * comparison['difference_exact_minus_shortcut']:+.2f} | [{100 * lo:+.2f}, {100 * hi:+.2f}] |")
    lines += ["", "| Direction | Distinct faces | jump1 shortcut / exact | jump4 shortcut / exact |", "|---|---|---|---|"]
    for name, direction in result["directions"].items():
        if not direction["photos"]:
            lines.append(f"| {name} | 0 | No eligible faces | No eligible faces |")
        else:
            values = [f"{direction['arms'][f'{p}/shortcut']['check_success']:.1%} / "
                      f"{direction['arms'][f'{p}/exact']['check_success']:.1%}" for p in ("jump1", "jump4")]
            lines.append(f"| {name} | {direction['photos']} | {values[0]} | {values[1]} |")
    lines += ["", "Photos are bootstrapped within smile-change direction, retaining all five seeds and all arms together.",
              "Pooled results weight faces equally. Numerical failures remain unsuccessful outcomes in the denominator.",
              "Finite corrupted edits are retained; no prompt, threshold or gain is tuned on these outcomes.",
              "score.json includes direction-specific intervals, conditional transfer, pixel/perceptual distances,",
              "target-directed cosine-margin changes, scale usage, timing and memory. Continuous metrics include observation counts.",
              "Cosine margins are not calibrated probabilities; evaluator transfer is not proof of visual realism.",
              "The two prespecified comparisons are reported separately without an automatic overall H2-support label."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.out_dir.exists() or args.boot < 1:
        p.error("choose a new score output directory and --boot >= 1")
    protocol, cohorts, results = load_results(args.runs)
    result = analyse(protocol, cohorts, results, args.boot, args.seed)
    args.out_dir.mkdir(parents=True)
    h2.write_json(args.out_dir / "score.json", result)
    text = markdown(result)
    (args.out_dir / "score.md").write_text(text)
    print(text)


if __name__ == "__main__":
    sys.exit(main())
