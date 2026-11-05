"""Run on the cluster: python3 collect_cluster.py ~/mfdime/results/h2_full_v1.

Writes a compact tar.gz beside the result directory; never changes experiment outputs.
Uses only Python's standard library. Selected images: first two frozen-cohort photos,
seed 0, all four arms. Selection never depends on generated outcomes.
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.runs.expanduser().resolve()
    out = args.out or root.parent / (root.name + "_assignment.tar.gz")
    if out.exists():
        parser.error(f"Output already exists: {out}; choose --out with a new filename")
    files = set()

    def require(path):
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path}")
        files.add(path)

    require(root / "protocol.json")
    for name in ("score.md", "score.json"):
        require(root / "score" / name)
    wrapper = json.loads((root / "protocol.json").read_text())
    protocol = wrapper["protocol"]
    for group in range(len(protocol["seed_groups"])):
        for name in ("env.json", "complete.json"):
            require(root / "jobs" / f"group{group}" / name)
    selections = []
    for source, target in protocol["pairs"]:
        pair = f"{source}_{target}"
        cohort_path = root / "cohorts" / f"{pair}.json"
        require(cohort_path)
        cohort = json.loads(cohort_path.read_text())
        for photo in cohort["names"][:2]:
            sample_hash = hashlib.sha256(photo.encode()).hexdigest()[:20]
            selections.append({"pair": pair, "photo": photo, "seed": 0,
                               "sample_hash": sample_hash})
            for peek, mode in protocol["arms"]:
                sample = root / "runs" / pair / "seed0" / peek / mode / "samples" / sample_hash
                row_path = sample / "manifest.jsonl"
                require(row_path)
                row = json.loads(row_path.read_text().strip())
                images = sorted((sample / "images").glob("*.png"))
                if row["status"] == "ok" and len(images) < 2:
                    raise SystemExit(f"Missing selected original/edited images: {sample}")
                files.update(images)
    # Preserve all small records for reanalysis, including per-sample provenance.
    # Omit per-step diagnostics and the bulk of generated images.
    for name in ("config.json", "manifest.jsonl", "identity.json", "complete.json"):
        files.update((root / "runs").rglob(name))
    repo = root.parent.parent
    sources = [repo / "README.md", repo / "pyproject.toml", repo / "uv.lock",
               repo / "jobs" / "h2_full.sbatch"]
    sources += sorted((repo / "src" / "mfdime").glob("*.py"))
    missing_sources = [str(path) for path in sources if not path.is_file()]
    if missing_sources:
        raise SystemExit(f"Missing source files: {missing_sources}")
    logs = sorted((repo / "logs").glob("mfdime-h2-full*"))
    selection_text = json.dumps({
        "selection_rule": "first two photos per frozen cohort; seed 0; all four arms",
        "examples": selections,
        "included_log_files": [p.name for p in logs if p.is_file()],
        "note": "Images illustrate outcomes; they are not a visual-quality evaluation. "
                "Source files are the cluster working copy at collection time; "
                "consult protocol.json for generation code identity."
    }, indent=2).encode()
    import io
    with tarfile.open(out, "x:gz") as archive:
        for path in sorted(files):
            archive.add(path, arcname=str(Path("experiment") / path.relative_to(root)))
        for path in sources:
            archive.add(path, arcname=str(Path("cluster_source") / path.relative_to(repo)))
        for path in logs:
            if path.is_file():
                archive.add(path, arcname=str(Path("logs") / path.name))
        info = tarfile.TarInfo("selection.json")
        info.size = len(selection_text)
        archive.addfile(info, io.BytesIO(selection_text))
    print(f"Created {out} ({out.stat().st_size / 1024**2:.1f} MiB)")
    print(f"Included {len(files)} experiment files and {len(selections)} matched examples.")
    if not logs:
        print("No matching Slurm logs found under repo/logs; collect them separately if stored elsewhere.")


if __name__ == "__main__":
    main()
