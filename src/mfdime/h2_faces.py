"""Full held-out CelebA-HQ/OpenCLIP H2 run: five single-GPU jobs, one seed each, batch size one.

All jobs share one frozen cohort, reuse the pilot's sampler and save per-image completion records for resume.
No batching benchmark, gradient normalisation or extra determinism settings are introduced.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from mfdime import celebahq, clip_smile, config, dime, explain, h2, h2_faces_pilot, imagenet, pmf, setups
from mfdime.common import env_info, predict, repo_sha, save_png

ARMS = (("jump1", "shortcut"), ("jump1", "exact"), ("jump4", "shortcut"), ("jump4", "exact"))
SEEDS = (0, 1, 2, 3, 4)
SKIP_TUNING = 64  # after the established 16 practice faces; also excluded by H1's face run
N_FACES = 512


def experiment_protocol(args, sched: dime.Schedule) -> dict:
    files = ("h2_faces.py", "clip_smile.py", "h2.py", "dime.py", "flow.py", "pmf.py", "celebahq.py",
             "explain.py", "h2_faces_pilot.py", "common.py", "setups.py", "config.py")
    sha = hashlib.sha256()
    for name in files:
        sha.update(name.encode())
        sha.update(Path(__file__).with_name(name).read_bytes())
    return {"schema": "faces_h2_v1", "code_sha256": sha.hexdigest(), "batch": 1, "n": args.n,
            "dataset": "celebahq", "split": "held_out", "practice_count": config.N_PRACTICE,
            "skip_tuning": SKIP_TUNING, "seeds": list(SEEDS), "arms": [list(a) for a in ARMS],
            "gain": config.GAIN["celebahq"], "noise_t": sched.noise_t, "grid": sched.grid, "scales": list(dime.SCALES),
            "clip_repo": clip_smile.REPO, "clip_revision": clip_smile.REVISION, "prompts": list(clip_smile.PROMPTS),
            "allow_tf32": True, "gradient_normalisation": False,
            "paths": {k: str(getattr(args, k)) for k in ("fire_repo", "pmf_ckpt", "steex_ckpt", "celebahq_zip",
                                                        "celeba_partition", "clip_dir")}}


def validate_cohort(record: dict, signature: str) -> None:
    if record.get("signature") != signature or record.get("cohort_sha256") != h2.digest(
            {k: v for k, v in record.items() if k != "cohort_sha256"}):
        raise ValueError("face cohort signature/hash mismatch")
    if record["split"] != "held_out" or record["names"] != [r["photo"] for r in record["selected"]]:
        raise ValueError("invalid face cohort ordering or split")
    if len(set(record["names"])) != len(record["names"]) or set(record["names"]) & set(record["excluded_names"]):
        raise ValueError("duplicate photos or practice/tuning leakage")
    eligible = [r for r in record["screened"] if r["label"] == r["guide_before"] == r["check_before"]]
    expected = [r | {"target": 1 - r["label"]} for r in eligible[:len(record["names"])]]
    if record["selected"] != expected:
        raise ValueError("cohort is not the first eligible original faces")
    if any(r["target"] != 1 - r["label"] for r in record["selected"]):
        raise ValueError("invalid smile targets")


def frozen_cohort(args, s, evaluator, signature: str) -> dict:
    path = args.out_dir / "cohort.json"
    if path.exists():
        record = h2.read_json(path)
        validate_cohort(record, signature)
        return record
    faces = celebahq.load_split(args.celebahq_zip, args.celeba_partition, "test")
    excluded = faces[:config.N_PRACTICE + SKIP_TUNING]
    candidates = faces[len(excluded):]
    selected, screened = [], []
    for start in range(0, len(candidates), 32):
        if path.exists():
            record = h2.read_json(path)
            validate_cohort(record, signature)
            return record
        block = candidates[start:start + 32]
        images = celebahq.load_images(args.celebahq_zip, [f.idx for f in block])
        guide = predict(s.guide, images, 1, args.device).tolist()
        check = h2_faces_pilot.scores(evaluator, images, 1)
        for i, face in enumerate(block):
            label = face.attrs["smile"]
            row = {"photo": f"{face.idx}.jpg", "idx": face.idx, "label": label, "guide_before": guide[i],
                   "check_before": int(check[i].argmax()), "smile_margin_before": float(check[i, 1] - check[i, 0])}
            screened.append(row)
            if row["guide_before"] == row["check_before"] == label:
                selected.append(row | {"target": 1 - label})
        if len(selected) >= args.n:
            break
    if len(selected) < args.n:
        raise ValueError(f"only {len(selected)} eligible held-out faces; requested {args.n}; no edits generated")
    record = {"signature": signature, "split": "held_out", "selection": "first eligible faces by HQ index",
              "excluded_names": [f"{f.idx}.jpg" for f in excluded], "practice_count": config.N_PRACTICE,
              "skip_tuning": SKIP_TUNING, "candidate_count": len(candidates), "screened": screened,
              "selected": selected[:args.n], "names": [r["photo"] for r in selected[:args.n]]}
    record["cohort_sha256"] = h2.digest(record)
    record = h2.publish_once(path, record)
    validate_cohort(record, signature)
    print(f"Frozen {len(record['names'])} held-out faces; excluded {len(excluded)} practice/tuning faces", flush=True)
    return record


def sample_identity(signature, cohort, entry, seed, peek, mode):
    return {"signature": signature, "cohort_sha256": cohort["cohort_sha256"], "photo": entry["photo"],
            "source": entry["label"], "target": entry["target"], "seed": seed, "peek": peek, "gradient_mode": mode}


def run_sample(args, s, evaluator, entry: dict, seed: int, sched: dime.Schedule, folder: Path, identity: dict) -> dict:
    row = h2.completed_sample(folder, identity)
    if row is not None:
        return row
    if folder.exists():
        archive = folder.parent.parent / "interrupted" / f"{folder.name}_{time.time_ns()}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        folder.rename(archive)
    (folder / "images").mkdir(parents=True)
    images = celebahq.load_images(args.celebahq_zip, [entry["idx"]])
    target = torch.tensor([entry["target"]])
    ph = setups.Photos([entry["photo"]], images, target)
    save_png(images[0], folder / "images" / f"{Path(entry['photo']).stem}_orig.png")
    started = time.perf_counter()
    try:
        rows, generated = explain.counterfactuals(args, s, ph, args.peek, [seed], sched, folder)
        row = rows[0] | {"status": "ok", "error": None}
    except FloatingPointError as error:
        row = {"photo": entry["photo"], "seed": seed, "peek": args.peek, "gradient_mode": args.gradient_mode,
               "target": entry["target"], "gain": args.gain, "status": "numerical_failure", "error": str(error),
               "guide_before": entry["guide_before"], "guide_after": None, "flipped": False, "scale": None,
               "seconds": None, "generator_calls": None, "batch_peak_allocated_bytes": None,
               "batch_peak_reserved_bytes": None, "failure_wall_seconds": time.perf_counter() - started}
    row.update({"dataset": "celebahq", "source": entry["label"], "label": entry["label"], "batch": 1,
                "check_before": entry["check_before"], "smile_margin_before": entry["smile_margin_before"]})
    if row["status"] == "ok":
        after = h2_faces_pilot.scores(evaluator, generated[0], 1)[0]
        with torch.no_grad():
            distance = float(s.perc(generated[0].to(args.device), images.to(args.device)))
        if not torch.isfinite(after).all() or not torch.isfinite(torch.tensor(distance)):
            raise FloatingPointError("nonfinite evaluator/measurement output; sample left incomplete")
        margin = float(after[1] - after[0])
        row.update({"check_after": int(after.argmax()), "check_flipped": bool(after.argmax() == target[0]),
                    "smile_margin_after": margin,
                    "target_margin_change": (margin - entry["smile_margin_before"]) * (1 if entry["target"] else -1),
                    "pixel_l1": float((generated[0] - images).abs().mean()), "perceptual": distance})
    else:
        row.update({"check_after": None, "check_flipped": False, "smile_margin_after": None,
                    "target_margin_change": None, "pixel_l1": None, "perceptual": None})
        print(f"numerical failure recorded: {entry['photo']}: {row['error']}", flush=True)
    (folder / "manifest.jsonl").write_text(json.dumps(row, allow_nan=False) + "\n")
    h2.write_json(folder / "complete.json", {"identity": identity, "row_sha256": h2.digest(row)})
    return row


def run_arm(args, s, evaluator, signature: str, cohort: dict, sched: dime.Schedule) -> None:
    out = args.out_dir / "runs" / f"seed{args.seed}" / args.peek / args.gradient_mode
    out.mkdir(parents=True, exist_ok=True)
    h2.write_json(out / "config.json", {"signature": signature, "batch": 1, "peek": args.peek,
                                       "gradient_mode": args.gradient_mode, "seed": args.seed})
    rows, originals, edited = [], [], []
    for i, entry in enumerate(cohort["selected"]):
        folder = h2.sample_dir(out, entry["photo"])
        identity = sample_identity(signature, cohort, entry, args.seed, args.peek, args.gradient_mode)
        row = run_sample(args, s, evaluator, entry, args.seed, sched, folder, identity)
        rows.append(row)
        if row["status"] == "ok":
            for collection, suffix in ((originals, "orig"), (edited, f"seed{args.seed}_cf")):
                with Image.open(folder / "images" / f"{Path(entry['photo']).stem}_{suffix}.png") as image:
                    collection.append(imagenet.to_tensor(image))
        if (i + 1) % 16 == 0 or i + 1 == len(cohort["selected"]):
            print(f"seed {args.seed} {args.peek}/{args.gradient_mode}: {i + 1}/{len(cohort['selected'])} complete", flush=True)
    (out / "manifest.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    if originals:
        explain.save_grids(torch.stack(originals), [torch.stack(edited)], out)
    h2.write_json(out / "complete.json", {"signature": signature, "cohort_sha256": cohort["cohort_sha256"],
                                         "rows_sha256": h2.digest(rows), "samples": len(rows)})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    setups.add_args(p)
    p.set_defaults(dataset="celebahq")
    p.add_argument("--clip-dir", type=Path, default=clip_smile.DIRECTORY)
    p.add_argument("--n", type=int, default=N_FACES)
    p.add_argument("--out-dir", type=Path, default=Path("results/h2_faces_full_v1"))
    p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = p.parse_args(argv)
    if args.dataset != "celebahq" or args.n < 1:
        p.error("full faces requires CelebA-HQ and --n >= 1")
    if repo_sha(args.fire_repo) != pmf.FIRE_SHA:
        p.error("initialise the pinned FiRe submodule")
    args.batch, args.gain, args.smoke = 1, config.GAIN["celebahq"], False
    args.evaluate, args.diagnostics = False, True  # evaluator applied afterward, exactly as in the pilot
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.source = args.target = None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sched = dime.dime_schedule()
    protocol = experiment_protocol(args, sched)
    signature = h2.digest(protocol)
    wrapper = {"signature": signature, "protocol": protocol}
    if h2.publish_once(args.out_dir / "protocol.json", wrapper) != wrapper:
        raise ValueError("existing face root uses different generation code/settings; choose a fresh output directory")
    job = args.out_dir / "jobs" / f"seed{args.seed}"
    job.mkdir(parents=True, exist_ok=True)
    with (job / "lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("this seed is already running") from error
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        print(f"load once: faces seed {args.seed}, batch 1", flush=True)
        s = setups.build(args)
        evaluator, info = clip_smile.load(args.clip_dir, args.device)
        h2.write_json(job / "env.json", env_info(args, s.info) | {"evaluator": info, "signature": signature})
        cohort = frozen_cohort(args, s, evaluator, signature)
        for peek, mode in ARMS:
            args.peek, args.gradient_mode = peek, mode
            run_arm(args, s, evaluator, signature, cohort, sched)
        h2.write_json(job / "complete.json", {"signature": signature, "seed": args.seed, "photos": len(cohort["names"])})
        print(f"faces seed {args.seed} complete", flush=True)


if __name__ == "__main__":
    sys.exit(main())
