"""Simple practice-only CelebA-HQ/OpenCLIP pilot: check smile labels, then run the four jump/gradient arms.

One GPU, one seed, at most four faces by default. Existing FiRe/STEEX/VGG assets are reused. The first 16 test
faces are the established practice split; no held-out photos or generated-image outcomes select prompts/faces.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch import Tensor, nn

from mfdime import clip_smile, config, dime, explain, h2_pilot, pmf, setups
from mfdime.common import env_info, repo_sha, save_grid, save_png


@torch.no_grad()
def scores(evaluator: nn.Module, images: Tensor, batch: int = 1) -> Tensor:
    return torch.cat([evaluator(images[i:i + batch]).cpu() for i in range(0, len(images), batch)])


def choose_faces(labels: list[int], guide: list[int], check: list[int], n: int) -> list[int]:
    """Prefer equal numbers of original smile/non-smile labels, among originals both models label correctly."""
    eligible = [i for i, label in enumerate(labels) if guide[i] == check[i] == label]
    chosen = []
    for label, quota in ((0, n // 2), (1, n - n // 2)):
        chosen.extend([i for i in eligible if labels[i] == label][:quota])
    chosen.extend(i for i in eligible if i not in chosen)
    return sorted(chosen[:n])


def write_evaluation(root: Path, pool: setups.Photos, record: dict, original_scores: Tensor, info: dict) -> dict:
    labels, guide = record["label"], record["guide_pred"]
    predicted = original_scores.argmax(1).tolist()
    result: dict = {"purpose": "practice-only smile evaluator check", "evaluator": info,
              "openclip": clip_smile.classification_report(labels, predicted),
              "guide": clip_smile.classification_report(labels, guide), "photos": []}
    for i, name in enumerate(pool.names):
        result["photos"].append({"photo": name, "label": labels[i], "guide": guide[i], "openclip": predicted[i],
                                 "neutral_cosine": float(original_scores[i, 0]), "smile_cosine": float(original_scores[i, 1]),
                                 "smile_margin": float(original_scores[i, 1] - original_scores[i, 0])})
    (root / "evaluator.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (root / "originals.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["photos"][0]))
        writer.writeheader()
        writer.writerows(result["photos"])
    save_grid([pool.x[i:i + 4] for i in range(0, len(pool.x), 4)], root / "originals.jpg")
    lines = ["# OpenCLIP smile check", "", f"{len(labels)} unfiltered practice originals; 0 = non-smiling, 1 = smiling.",
             "Scores are cosine similarities, not calibrated probabilities. Fixed prompts:", "",
             *[f"- {i}: {prompt}" for i, prompt in enumerate(info["prompts"])], "",
             "| Model | Accuracy | Non-smiling recall (n) | Smiling recall (n) | Balanced accuracy |", "|---|---|---|---|---|"]
    for name in ("guide", "openclip"):
        metric = result[name]
        recalls = [f"{c['recall']:.1%} ({c['count']})" if c["recall"] is not None else "not represented (0)"
                   for c in metric["classes"].values()]
        balanced = f"{metric['balanced_accuracy']:.1%}" if metric["balanced_accuracy"] is not None else "N/A"
        lines.append(f"| {name} | {metric['accuracy']:.1%} | {recalls[0]} | {recalls[1]} | {balanced} |")
    lines += ["", "The confusion matrix and each original's labels/margins are in evaluator.json and originals.csv.",
              "This small practice check cannot establish reliable smile-recognition accuracy on the full dataset."]
    (root / "evaluator.md").write_text("\n".join(lines) + "\n")
    return result


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    setups.add_args(p)
    p.set_defaults(dataset="celebahq")
    p.add_argument("--clip-dir", type=Path, default=clip_smile.DIRECTORY)
    p.add_argument("--out-dir", type=Path, default=Path("results/h2_faces_pilot_v1"))
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.dataset != "celebahq" or not 1 <= args.n <= config.N_PRACTICE:
        p.error(f"pilot requires CelebA-HQ and 1 <= --n <= {config.N_PRACTICE}")
    if repo_sha(args.fire_repo) != pmf.FIRE_SHA:
        p.error(f"initialise the pinned FiRe submodule ({pmf.FIRE_SHA})")
    if args.out_dir.exists():
        p.error("output already exists; choose a new --out-dir")
    args.batch, args.gain, args.smoke = 1, config.GAIN["celebahq"], False
    args.source = args.target = None  # face source/target classes are per photo, not ImageNet label IDs
    # Generation uses the existing loop. OpenCLIP is applied only afterward by this pilot, never in guidance.
    args.evaluate, args.diagnostics = False, True
    args.split, args.skip, args.seeds = "practice", 0, [args.seed]
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    print("load face generator and guide", flush=True)
    s = setups.build(args)
    print("load frozen OpenCLIP evaluator", flush=True)
    check, info = clip_smile.load(args.clip_dir, args.device)
    (args.out_dir / "env.json").write_text(json.dumps(env_info(args, s.info) | {"evaluator": info}, indent=2))
    pool, record = setups.photos(args, s, "practice", 0)
    original_scores = scores(check, pool.x)
    evaluation = write_evaluation(args.out_dir, pool, record, original_scores, info)
    chosen = choose_faces(record["label"], record["guide_pred"], original_scores.argmax(1).tolist(), args.n)
    selected_record = record | {"practice_names": pool.names, "names": [pool.names[i] for i in chosen],
                                "practice_labels": record["label"], "label": [record["label"][i] for i in chosen],
                                "guide_pred": [record["guide_pred"][i] for i in chosen],
                                "targets": pool.target[chosen].tolist(),
                                "selected_practice_indices": chosen, "requested": args.n,
                                "selection": "prefer equal classes among guide/OpenCLIP/annotation agreement"}
    (args.out_dir / "photos.json").write_text(json.dumps(selected_record, indent=2) + "\n")
    print((args.out_dir / "evaluator.md").read_text(), flush=True)
    if not chosen:
        raise SystemExit("STOP: no practice faces have correct guide and OpenCLIP labels; evaluator report was saved")
    if len(chosen) < args.n or len({record["label"][i] for i in chosen}) < 2:
        print("WARNING: fewer eligible faces or only one original smile class; inspect photos.json", flush=True)
    ph = setups.Photos([pool.names[i] for i in chosen], pool.x[chosen], pool.target[chosen])
    selected_before = original_scores[chosen]
    sched, results = dime.dime_schedule(), {}
    for peek, mode in config.H2_ARMS:
        args.peek, args.gradient_mode = peek, mode
        name = f"{peek}/{mode}"
        out = args.out_dir / name
        (out / "images").mkdir(parents=True)
        cfg = {k: str(v) for k, v in vars(args).items()} | {"evaluator": info, "steps": len(sched.grid) - 1,
               "noise_t": sched.noise_t, "grid": sched.grid, "scales": dime.SCALES, "gradient_normalisation": False}
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        for i, photo in enumerate(ph.names):
            save_png(ph.x[i], out / "images" / f"{Path(photo).stem}_orig.png")
        rows, images = explain.counterfactuals(args, s, ph, peek, args.seeds, sched, out)
        after = scores(check, images[0])
        with torch.no_grad():
            for i, row in enumerate(rows):
                before_margin = float(selected_before[i, 1] - selected_before[i, 0])
                after_margin = float(after[i, 1] - after[i, 0])
                row.update({"dataset": "celebahq", "source": record["label"][chosen[i]],
                            "label": record["label"][chosen[i]], "check_before": int(selected_before[i].argmax()),
                            "check_after": int(after[i].argmax()), "check_flipped": bool(after[i].argmax() == ph.target[i]),
                            "smile_margin_before": before_margin, "smile_margin_after": after_margin,
                            "target_margin_change": (after_margin - before_margin) * (1 if int(ph.target[i]) else -1),
                            "pixel_l1": float((images[0][i] - ph.x[i]).abs().mean()),
                            "perceptual": float(s.perc(images[0][i:i + 1].to(args.device), ph.x[i:i + 1].to(args.device)))})
        (out / "manifest.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        explain.save_grids(ph.x, images, out)
        results[name] = rows
    summary = h2_pilot.report(args.out_dir, results)
    summary["evaluator_on_practice_originals"] = evaluation["openclip"]
    for name, rows in results.items():
        summary["arms"][name]["by_target"] = {str(target): {"n": len(group),
            "check_success": sum(r["check_flipped"] for r in group) / len(group)}
            for target in (0, 1) if (group := [r for r in rows if r["target"] == target])}
        summary["arms"][name]["mean_target_margin_change"] = sum(r["target_margin_change"] for r in rows) / len(rows)
    (args.out_dir / "pilot.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (args.out_dir / "pilot.md").open("a") as stream:
        stream.write("\nSee evaluator.md for the unfiltered original-image smile check. Target 1 adds a smile; target 0 removes it.\n")
        stream.write("OpenCLIP predictions use uint8 RGB quantisation and the checkpoint's own evaluation transform.\n")
    (args.out_dir / "complete.json").write_text(json.dumps({"photos": ph.names, "seed": args.seed, "arms": list(results)}) + "\n")
    print((args.out_dir / "pilot.md").read_text(), flush=True)


if __name__ == "__main__":
    sys.exit(main())
