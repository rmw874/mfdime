"""Run H2's four arms on one ImageNet class pair's practice photos, with paired noise and full diagnostics.

PYTHONPATH=src python -P -m mfdime.h2_pilot --source 340 --target 339 --out-dir runs/h2_pilot/340_339
This is a feasibility pilot, not a held-out hypothesis test. --smoke uses random models and fake data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mfdime import config, dime, dmf, explain, setups
from mfdime.common import predict, repo_sha, save_png

ARMS = tuple((peek, mode) for peek in ("jump1", "jump4") for mode in ("shortcut", "exact"))


def cohort(args, s: setups.Setup) -> tuple[setups.Photos, dict]:
    """Filter *within* the established practice split, never shifting the practice/held-out boundary."""
    if s.check is None:
        raise ValueError("H2 requires an independent classifier")
    pool, record = setups.photos(args, s, "practice", 0)
    check_pred = predict(s.check, pool.x, args.batch, args.device)
    eligible = [i for i, c in enumerate(check_pred.tolist()) if args.smoke or c == args.source]
    chosen = eligible[:args.n]
    record.update({"requested": args.n, "eligible": len(eligible), "selected": len(chosen),
                   "practice_names": pool.names, "check_predictions": check_pred.tolist(),
                   "smoke_bypasses_correctness_filter": args.smoke,
                   "names": [pool.names[i] for i in chosen]})
    if not chosen:
        raise SystemExit("STOP: no practice photos recognised as the source by both classifiers")
    if len(chosen) < args.n:
        print(f"WARNING: only {len(chosen)} of {args.n} requested practice photos are eligible", flush=True)
    return setups.Photos(record["names"], pool.x[chosen], pool.target[chosen]), record


def report(root: Path, results: dict[str, list[dict]]) -> dict:
    """A descriptive pilot report; no significance or final H2 claim from practice images."""
    summary: dict = {"purpose": "practice-only feasibility pilot", "arms": {}, "comparisons": {}}
    lines = ["# H2 pilot", "", "Practice photos only; these results do not establish H2.", "",
             "| arm | n | guide success | evaluator success | joint success | mean seconds | peak allocated GiB |",
             "|---|---|---|---|---|---|---|"]
    for name, rows in results.items():
        def mean(key: str, rows=rows) -> float:
            return sum(float(r[key]) for r in rows) / len(rows)

        diagnostics = [json.loads(line) for line in (root / name / "diagnostics.jsonl").read_text().splitlines()]
        if not diagnostics or not all(r["finite"] for r in diagnostics):
            raise ValueError(f"missing or nonfinite diagnostics for {name}")
        peaks = [r["batch_peak_allocated_bytes"] for r in rows if r["batch_peak_allocated_bytes"] is not None]
        used_scales = {str(scale): sum(r["scale"] == scale for r in rows) for scale in (*dime.SCALES, None)}
        arm = {"explanations": len(rows), "guide_success": mean("flipped"), "check_success": mean("check_flipped"),
               "joint_success": sum(r["flipped"] and r["check_flipped"] for r in rows) / len(rows),
               "seconds": mean("seconds"), "pixel_l1": mean("pixel_l1"), "perceptual": mean("perceptual"),
               "peak_allocated_gib": max(peaks) / 2**30 if peaks else None, "scale_counts": used_scales,
               "all_steps_finite": True, "trace_records": len(diagnostics)}
        for metric in ("loss_gradient_l2", "guidance_l2", "update_l2"):
            values = [v for r in diagnostics for v in r[metric]]
            arm[metric] = {"mean": sum(values) / len(values), "max": max(values)}
        summary["arms"][name] = arm
        peak_str = f"{arm['peak_allocated_gib']:.2f}" if peaks else "CPU"
        lines.append(f"| {name} | {len(rows)} | {arm['guide_success']:.1%} | {arm['check_success']:.1%} | "
                     f"{arm['joint_success']:.1%} | {arm['seconds']:.2f} | {peak_str} |")
    for peek in ("jump1", "jump4"):
        a, b = (results[f"{peek}/{mode}"] for mode in ("shortcut", "exact"))
        by_key = [{(r["photo"], r["seed"]): r for r in rows} for rows in (a, b)]
        if len(by_key[0]) != len(a) or len(by_key[1]) != len(b) or by_key[0].keys() != by_key[1].keys():
            raise ValueError(f"unpaired pilot results for {peek}")
        delta = sum(int(by_key[1][k]["check_flipped"]) - int(by_key[0][k]["check_flipped"]) for k in by_key[0]) / len(a)
        summary["comparisons"][peek] = {"check_success_difference_exact_minus_shortcut": delta}
        lines += ["", f"{peek}: exact minus shortcut evaluator success = {100 * delta:+.1f} percentage points."]
    lines += ["", "Inspect diagnostics.jsonl and each arm's grid_00.jpg before scheduling held-out experiments.",
              "Generator-call counts include forward calls only; backward cost is reflected in time and memory.",
              "Timing covers guidance and final decoding, including diagnostic logging, but excludes evaluator scoring."]
    (root / "pilot.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (root / "pilot.md").write_text("\n".join(lines) + "\n")
    return summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    setups.add_args(p)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n", type=int, default=4, help="eligible practice photos (at most 16)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--gain", type=float, default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.dataset != "imagenet" or not 1 <= args.n <= config.N_PRACTICE or args.batch < 1:
        p.error(f"pilot requires ImageNet, 1 <= --n <= {config.N_PRACTICE}, and --batch >= 1")
    if not args.smoke and repo_sha(args.dmf_repo) != dmf.DMF_SHA:
        p.error(f"initialise the pinned DMF submodule ({dmf.DMF_SHA}) before running the pilot")
    args.peek, args.gradient_mode = "jump1", "shortcut"
    args.evaluate = args.diagnostics = True
    args.split, args.skip, args.seeds = "practice", 0, [args.seed]
    s, sched = explain.prepare(args)
    root = args.out_dir
    ph, record = cohort(args, s)
    (root / "photos.json").write_text(json.dumps(record, indent=2) + "\n")
    results = {}
    for peek, mode in ARMS:
        args.peek, args.gradient_mode = peek, mode
        name = f"{peek}/{mode}"
        out = root / name
        out.mkdir(parents=True)
        cfg = {k: str(v) for k, v in vars(args).items()} | {"steps": len(sched.grid) - 1,
               "t_noise": sched.noise_t, "t_first_step": sched.grid[0], "scales": dime.SCALES,
               "out_dir": str(out), "pilot_root": str(root),
               "evaluator": "random weights" if args.smoke else "ViT_B_16_Weights.IMAGENET1K_V1",
               "gradient_normalisation": False}
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        (out / "photos.json").write_text(json.dumps(record, indent=2) + "\n")
        (out / "images").mkdir()
        for i, photo in enumerate(ph.names):
            save_png(ph.x[i], out / "images" / f"{Path(photo).stem}_orig.png")
        rows, images = explain.counterfactuals(args, s, ph, peek, args.seeds, sched, out)
        explain.save_grids(ph.x, images, out)
        results[name] = rows
    report(root, results)
    # Written only after all four arms, grids, diagnostics, and report have succeeded.
    (root / "complete.json").write_text(json.dumps({"arms": list(results), "photos": ph.names, "seed": args.seed}) + "\n")
    print((root / "pilot.md").read_text(), flush=True)


if __name__ == "__main__":
    sys.exit(main())
