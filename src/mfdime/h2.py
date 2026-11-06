"""Full held-out H2 experiment. Three single-GPU jobs share a frozen cohort and split seeds as 0/1, 2/3, and 4.

PYTHONPATH=src python -P -m mfdime.h2 --group 0 --out-dir results/h2_full_v1
Completed samples are reused on resubmission; interrupted samples are archived and retried. Numerical failures
are recorded as unsuccessful outcomes, never dropped. No determinism or gradient-normalisation settings are added.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image

from mfdime import config, dime, dmf, explain, imagenet, setups
from mfdime.common import env_info, predict, repo_sha, save_png


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    """Atomic replacement, so interruption never publishes half a completion record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def publish_once(path: Path, value):
    """First publisher wins atomically. Concurrent jobs all adopt the same complete record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass
        return read_json(path)
    finally:
        tmp.unlink(missing_ok=True)


def experiment_protocol(args, sched: dime.Schedule) -> dict:
    sources = ("h2.py", "config.py", "dime.py", "flow.py", "explain.py", "setups.py", "dmf.py", "imagenet.py", "common.py")
    source_hash = hashlib.sha256()
    for name in sources:
        source_hash.update(name.encode())
        source_hash.update(Path(__file__).with_name(name).read_bytes())
    return {"schema": 1, "dataset": "imagenet", "split": "held_out", "practice_count": config.N_PRACTICE,
            "pairs": [list(p) for p in config.H2_PAIRS], "arms": [list(a) for a in config.H2_ARMS],
            "seeds": [s for group in config.H2_SEED_GROUPS for s in group],
            "seed_groups": [list(group) for group in config.H2_SEED_GROUPS], "batch": 1, "gain": config.GAIN["imagenet"],
            "scales": list(dime.SCALES), "noise_t": sched.noise_t, "grid": sched.grid,
            "gradient_normalisation": False, "allow_tf32": True, "code_sha256": source_hash.hexdigest(),
            "guide": "ResNet50_Weights.IMAGENET1K_V1", "evaluator": "ViT_B_16_Weights.IMAGENET1K_V1",
            "model_paths": {k: str(getattr(args, k)) for k in ("dmf_repo", "dmf_ckpt", "sit_ckpt", "vae", "imagenet_val")}}


@torch.no_grad()
def frozen_photos(args, s: setups.Setup, signature: str) -> tuple[setups.Photos, dict]:
    """Freeze source-correct split before evaluator filtering; reuse identities without reselecting on later jobs."""
    if s.check is None:
        raise ValueError("H2 requires an independent evaluator")
    path = args.out_dir / "cohorts" / f"{args.source}_{args.target}.json"
    items = imagenet.load_classes(args.imagenet_val, [args.source, args.target])[args.source]
    if not items or len({name for name, _ in items}) != len(items):
        raise ValueError("empty source dataset or duplicate photo names")
    names, images = [name for name, _ in items], torch.stack([x for _, x in items])
    if path.exists():
        record = read_json(path)
    else:
        guide = predict(s.guide, images, config.BATCH, args.device)
        correct = [i for i, c in enumerate(guide.tolist()) if c == args.source]
        practice, held = correct[:config.N_PRACTICE], correct[config.N_PRACTICE:]
        check = predict(s.check, images[held], args.batch, args.device).tolist() if held else []
        selected = [names[i] for i, c in zip(held, check, strict=True) if c == args.source]
        candidate = {"signature": signature, "source": args.source, "target": args.target, "split": "held_out",
                     "candidates": len(items), "guide_correct": len(correct), "practice_names": [names[i] for i in practice],
                     "held_out_names": [names[i] for i in held], "check_predictions": check, "names": selected,
                     "eligible": len(selected)}
        candidate["cohort_sha256"] = digest(candidate)
        record = publish_once(path, candidate)
    validate_cohort(record, signature, args.source, args.target)
    lookup = {name: i for i, name in enumerate(names)}
    if any(name not in lookup for name in record["names"]):
        raise ValueError("frozen cohort contains photos absent from the current dataset")
    selected_images = images[[lookup[name] for name in record["names"]]]
    print(f"{args.source}->{args.target}: {record['eligible']} eligible held-out photos "
          f"of {len(record['held_out_names'])} guide-correct candidates", flush=True)
    return setups.Photos(record["names"], selected_images, torch.full((len(selected_images),), args.target)), record


def validate_cohort(record: dict, signature: str, source: int, target: int) -> None:
    body = {k: v for k, v in record.items() if k != "cohort_sha256"}
    if record.get("cohort_sha256") != digest(body) or record.get("signature") != signature:
        raise ValueError("cohort identity does not match this experiment")
    if (record["source"], record["target"], record["split"]) != (source, target, "held_out"):
        raise ValueError("cohort task or split mismatch")
    if len(set(record["names"])) != len(record["names"]) or record["eligible"] != len(record["names"]):
        raise ValueError("duplicate or inconsistent cohort identities")
    if set(record["practice_names"]) & set(record["held_out_names"]) or not set(record["names"]) <= set(record["held_out_names"]):
        raise ValueError("practice/held-out leakage or invalid cohort selection")
    if len(record["held_out_names"]) != len(record["check_predictions"]):
        raise ValueError("cohort evaluator predictions are incomplete")
    expected = [name for name, pred in zip(record["held_out_names"], record["check_predictions"], strict=True) if pred == source]
    if record["names"] != expected:
        raise ValueError("cohort is not the complete evaluator-correct held-out set")


def sample_dir(arm_dir: Path, photo: str) -> Path:
    # Hash the whole dataset identity: identical stems from different files must not collide.
    return arm_dir / "samples" / hashlib.sha256(photo.encode()).hexdigest()[:20]


def completed_sample(folder: Path, identity: dict) -> dict | None:
    marker = folder / "complete.json"
    if not marker.exists():
        return None
    complete = read_json(marker)
    if complete["identity"] != identity:
        raise ValueError(f"completed sample belongs to another experiment: {folder}")
    rows = [json.loads(line) for line in (folder / "manifest.jsonl").read_text().splitlines()]
    if len(rows) != 1 or complete["row_sha256"] != digest(rows[0]):
        raise ValueError(f"damaged completed sample: {folder}")
    row = rows[0]
    if row["status"] == "ok":
        image = folder / "images" / f"{Path(identity['photo']).stem}_seed{identity['seed']}_cf.png"
        if not image.is_file():
            raise ValueError(f"completed sample image is missing: {image}")
    elif row["status"] != "numerical_failure":
        raise ValueError(f"unknown sample status: {row['status']}")
    return row


def run_sample(args, s: setups.Setup, ph: setups.Photos, seed: int, sched: dime.Schedule,
               folder: Path, identity: dict) -> dict:
    existing = completed_sample(folder, identity)
    if existing is not None:
        print(f"resume: {ph.names[0]} seed {seed} {args.peek}/{args.gradient_mode}", flush=True)
        return existing
    if folder.exists():
        # Keep interrupted attempts for inspection; never delete an output or append duplicate rows.
        archive = folder.parent.parent / "interrupted" / f"{folder.name}_{time.time_ns()}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        folder.rename(archive)
    folder.mkdir(parents=True)
    (folder / "images").mkdir()
    save_png(ph.x[0], folder / "images" / f"{Path(ph.names[0]).stem}_orig.png")
    write_json(folder / "identity.json", identity)
    started = time.perf_counter()
    try:
        rows, _ = explain.counterfactuals(args, s, ph, args.peek, [seed], sched, folder)
        row = rows[0] | {"status": "ok"}
    except FloatingPointError as error:
        # This is an outcome of the method, not a reason to remove the original from the denominator.
        row = {"photo": ph.names[0], "source": args.source, "target": args.target, "seed": seed,
               "peek": args.peek, "gradient_mode": args.gradient_mode, "gain": args.gain,
               "status": "numerical_failure", "error": str(error), "flipped": False, "check_flipped": False,
               "guide_before": None, "guide_after": None, "check_before": None, "check_after": None,
               "scale": None, "pixel_l1": None, "perceptual": None, "seconds": None, "generator_calls": None,
               "batch_peak_allocated_bytes": None, "batch_peak_reserved_bytes": None,
               "failure_wall_seconds": time.perf_counter() - started}
        print(f"numerical failure recorded: {ph.names[0]}: {error}", flush=True)
    # Other errors (OOM, filesystem, missing assets, etc.) stop the job, leaving a resumable incomplete attempt.
    (folder / "manifest.jsonl").write_text(json.dumps(row, allow_nan=False) + "\n")
    write_json(folder / "complete.json", {"identity": identity, "row_sha256": digest(row)})
    return row


def run_arm(args, s: setups.Setup, ph: setups.Photos, record: dict, seed: int, sched: dime.Schedule,
            signature: str) -> None:
    arm_dir = args.out_dir / "runs" / f"{args.source}_{args.target}" / f"seed{seed}" / args.peek / args.gradient_mode
    arm_dir.mkdir(parents=True, exist_ok=True)
    write_json(arm_dir / "config.json", {"signature": signature, "cohort_sha256": record["cohort_sha256"],
               "source": args.source, "target": args.target, "seed": seed, "peek": args.peek,
               "gradient_mode": args.gradient_mode, "batch": 1, "gain": args.gain, "split": "held_out"})
    rows, originals, counterfactuals = [], [], []
    for i, photo in enumerate(ph.names):
        identity = {"signature": signature, "cohort_sha256": record["cohort_sha256"], "source": args.source,
                    "target": args.target, "photo": photo, "seed": seed, "peek": args.peek, "gradient_mode": args.gradient_mode}
        folder = sample_dir(arm_dir, photo)
        row = run_sample(args, s, setups.Photos([photo], ph.x[i:i + 1], ph.target[i:i + 1]), seed, sched, folder, identity)
        rows.append(row)
        if row["status"] == "ok":
            with Image.open(folder / "images" / f"{Path(photo).stem}_seed{seed}_cf.png") as image:
                counterfactuals.append(imagenet.to_tensor(image))
            originals.append(ph.x[i])
    (arm_dir / "manifest.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    if originals:
        explain.save_grids(torch.stack(originals), [torch.stack(counterfactuals)], arm_dir)
    write_json(arm_dir / "complete.json", {"signature": signature, "cohort_sha256": record["cohort_sha256"],
               "rows_sha256": digest(rows), "samples": len(rows), "grid_excludes_numerical_failures": True})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    setups.add_args(p)
    p.add_argument("--group", type=int, choices=range(len(config.H2_SEED_GROUPS)), required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results/h2_full_v1"))
    args = p.parse_args(argv)
    if args.dataset != "imagenet":
        p.error("the full H2 experiment currently supports ImageNet only")
    if repo_sha(args.dmf_repo) != dmf.DMF_SHA:
        p.error(f"initialise the pinned DMF submodule ({dmf.DMF_SHA})")
    args.batch, args.gain = 1, config.GAIN["imagenet"]
    args.smoke, args.evaluate, args.diagnostics = False, True, True
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sched = dime.dime_schedule()
    protocol = experiment_protocol(args, sched)
    signature = digest(protocol)
    existing = publish_once(args.out_dir / "protocol.json", {"signature": signature, "protocol": protocol})
    if existing != {"signature": signature, "protocol": protocol}:
        raise SystemExit("STOP: protocol/code/paths differ from this output root; use a fresh --out-dir")
    job_dir = args.out_dir / "jobs" / f"group{args.group}"
    job_dir.mkdir(parents=True, exist_ok=True)
    with (job_dir / "lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("STOP: this seed group is already running") from error
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        print(f"load once: group {args.group}, seeds {config.H2_SEED_GROUPS[args.group]}", flush=True)
        s = setups.build(args)
        write_json(job_dir / "env.json", env_info(args, s.info) | {"signature": signature})
        total = 0
        for source, target in config.H2_PAIRS:
            args.source, args.target = source, target
            ph, record = frozen_photos(args, s, signature)
            total += len(ph.names)
            for seed in config.H2_SEED_GROUPS[args.group]:
                for peek, mode in config.H2_ARMS:
                    args.peek, args.gradient_mode = peek, mode
                    run_arm(args, s, ph, record, seed, sched, signature)
        if total == 0:
            raise SystemExit("STOP: no eligible held-out photos across the three pairs")
        write_json(job_dir / "complete.json", {"signature": signature, "seeds": list(config.H2_SEED_GROUPS[args.group]),
                   "photos": total, "explanations": total * len(config.H2_SEED_GROUPS[args.group]) * len(config.H2_ARMS)})
        print(f"group {args.group} complete", flush=True)


if __name__ == "__main__":
    sys.exit(main())
