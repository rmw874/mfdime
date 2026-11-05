"""Complete setup-node downloads and offline checks for the CelebA-HQ/OpenCLIP pilot.

PYTHONPATH=src uv run --group faces python -P -m mfdime.setup_faces_pilot
Use --check-only to report every missing/invalid prerequisite without downloads. Existing completed assets are reused.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torchvision.models import VGG19_Weights

from mfdime import celebahq, clip_smile, config, download_clip_smile, pmf
from mfdime.common import repo_sha

FIRE_REPO = "DapaoZeng/FiRe"
HQ_REPO, HQ_REVISION = "liusq/CelebAMask-HQ", "9cd422d28a64c1f5e7144f3a5a398b78bbb9ad11"
PARTITION_REPO, PARTITION_REVISION = "noname110/celeba", "438b96fa79375870b726cd408a3da1eda4f074c4"
STEEX_URL = "https://github.com/valeoai/STEEX/releases/download/v.0.1/checkpoints_decision_densenet.tar.gz"
STEEX_ARCHIVE_BYTES = 229725745  # official v.0.1 release metadata
STEEX_MEMBER = "decision_densenet/celebamaskhq/checkpoint.tar"


def present(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def required_files() -> list[Path]:
    return [config.FIRE_REPO / "src/pmf/dit.py", config.PMF_CKPT, config.STEEX_CKPT, config.CELEBAHQ_ZIP,
            config.CELEBA_PARTITION, Path(torch.hub.get_dir()) / "checkpoints/vgg19-dcbb9e9d.pth",
            clip_smile.DIRECTORY / clip_smile.WEIGHT_FILE, clip_smile.DIRECTORY / "open_clip_config.json",
            clip_smile.DIRECTORY / "download.json"]


def audit() -> list[str]:
    """Offline preflight: collect all missing paths and structural problems in one pass."""
    issues = [f"missing or empty: {path}" for path in required_files() if not present(path)]
    if importlib.util.find_spec("open_clip") is None:
        issues.append("OpenCLIP package missing: run uv sync --frozen --group faces")
    if present(config.FIRE_REPO / "src/pmf/dit.py") and repo_sha(config.FIRE_REPO) != pmf.FIRE_SHA:
        issues.append(f"FiRe submodule must be at {pmf.FIRE_SHA}: git submodule update --init third_party/fire")
    metadata = clip_smile.DIRECTORY / "download.json"
    if present(metadata):
        try:
            data = json.loads(metadata.read_text())
            if data.get("repo") != clip_smile.REPO or data.get("revision") != clip_smile.REVISION:
                issues.append("OpenCLIP checkpoint revision mismatch; run mfdime.download_clip_smile")
        except (OSError, ValueError) as error:
            issues.append(f"invalid OpenCLIP download metadata: {error}")
    if present(config.CELEBAHQ_ZIP) and present(config.CELEBA_PARTITION):
        try:
            faces = celebahq.load_split(config.CELEBAHQ_ZIP, config.CELEBA_PARTITION, "test")
            if len(faces) < config.N_PRACTICE:
                issues.append(f"dataset has only {len(faces)} test faces, expected at least {config.N_PRACTICE}")
            with zipfile.ZipFile(config.CELEBAHQ_ZIP) as archive:
                for face in faces[:config.N_PRACTICE]:
                    archive.getinfo(f"{celebahq.ROOT}CelebA-HQ-img/{face.idx}.jpg")
        except (OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile) as error:
            issues.append(f"invalid CelebA-HQ zip/partition layout: {error}")
    return issues


def download_hf(repo: str, revision: str, repo_type: str, filename: str, destination: Path) -> None:
    if present(destination):
        print(f"reuse: {destination}", flush=True)
        return
    print(f"download: {repo}/{filename} -> {destination}", flush=True)
    # These exact root-level filenames were checked against the pinned repositories' file lists.
    path = Path(hf_hub_download(repo_id=repo, revision=revision, repo_type=repo_type,
                               filename=filename, local_dir=destination.parent))
    if path.resolve() != destination.resolve():
        raise ValueError(f"unexpected Hugging Face download path: {path}; expected {destination}")
    if not present(destination):
        raise OSError(f"download is empty or missing: {destination}")


def download_steex_archive(destination: Path) -> None:
    if present(destination) and destination.stat().st_size == STEEX_ARCHIVE_BYTES:
        print(f"reuse: {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    print(f"download: STEEX decision classifiers (230 MB) -> {destination}", flush=True)
    request = urllib.request.Request(STEEX_URL, headers={"User-Agent": "mfdime-face-pilot-setup"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)
    if partial.stat().st_size != STEEX_ARCHIVE_BYTES:
        raise OSError(f"incomplete STEEX archive: {partial.stat().st_size}/{STEEX_ARCHIVE_BYTES} bytes")
    os.replace(partial, destination)


def extract_steex(archive_path: Path, destination: Path) -> None:
    """Copy only the required regular-file member to its explicit destination, ignoring all other archive paths."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        matches = [member for member in archive.getmembers() if member.isfile()
                   and (member.name == STEEX_MEMBER or member.name.endswith("/" + STEEX_MEMBER))]
        if len(matches) != 1:
            raise ValueError(f"expected one {STEEX_MEMBER} in {archive_path}, found {len(matches)}")
        source = archive.extractfile(matches[0])
        if source is None:
            raise ValueError("could not read STEEX checkpoint from archive")
        with source, tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            tmp = Path(stream.name)
            try:
                shutil.copyfileobj(source, stream)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        try:
            if tmp.stat().st_size != matches[0].size or not present(tmp):
                raise OSError("incomplete extracted STEEX checkpoint")
            os.replace(tmp, destination)
        finally:
            tmp.unlink(missing_ok=True)
    print(f"extracted: {destination}", flush=True)


def setup_steex() -> None:
    if present(config.STEEX_CKPT):
        print(f"reuse: {config.STEEX_CKPT}", flush=True)
        return
    archive_path = config.STEEX_CKPT.parents[2] / "checkpoints_decision_densenet.tar.gz"
    download_steex_archive(archive_path)
    extract_steex(archive_path, config.STEEX_CKPT)


def setup_fire() -> None:
    if not present(config.FIRE_REPO / "src/pmf/dit.py"):
        subprocess.run(["git", "submodule", "update", "--init", str(config.FIRE_REPO)], check=True)
    if repo_sha(config.FIRE_REPO) != pmf.FIRE_SHA:
        raise ValueError(f"FiRe submodule must be at {pmf.FIRE_SHA}; run git submodule update --init third_party/fire")


def setup_vgg() -> None:
    path = Path(torch.hub.get_dir()) / "checkpoints/vgg19-dcbb9e9d.pth"
    if present(path):
        print(f"reuse: {path}", flush=True)
        return
    print("download: VGG19 perceptual weights", flush=True)
    state = torch.hub.load_state_dict_from_url(VGG19_Weights.IMAGENET1K_V1.url, map_location="cpu", check_hash=True)
    del state


def setup_clip() -> None:
    paths = [clip_smile.DIRECTORY / filename for filename in
             (clip_smile.WEIGHT_FILE, "open_clip_config.json", "download.json")]
    if all(present(path) for path in paths):
        try:
            metadata = json.loads(paths[2].read_text())
        except (OSError, ValueError):
            metadata = {}
        if metadata.get("repo") == clip_smile.REPO and metadata.get("revision") == clip_smile.REVISION:
            print(f"reuse: {clip_smile.DIRECTORY}", flush=True)
            return
    download_clip_smile.main([])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check-only", action="store_true", help="offline check of all required assets; no downloads")
    args = p.parse_args(argv)
    errors = []
    if not args.check_only:
        # Independent tasks continue after a failed download, so setup reports the complete set of problems.
        tasks = [
            ("FiRe submodule", setup_fire),
            ("pMF smile checkpoint", lambda: download_hf(FIRE_REPO, pmf.WEIGHTS_REV, "model", config.PMF_CKPT.name, config.PMF_CKPT)),
            ("STEEX classifier", setup_steex),
            ("CelebAMask-HQ zip", lambda: download_hf(HQ_REPO, HQ_REVISION, "dataset", config.CELEBAHQ_ZIP.name, config.CELEBAHQ_ZIP)),
            ("CelebA partition", lambda: download_hf(PARTITION_REPO, PARTITION_REVISION, "dataset",
                                                   config.CELEBA_PARTITION.name, config.CELEBA_PARTITION)),
            ("VGG19", setup_vgg),
            ("OpenCLIP", setup_clip),
        ]
        for label, task in tasks:
            try:
                task()
            except Exception as error:
                errors.append(f"{label}: {type(error).__name__}: {error}")
                print(f"SETUP ERROR: {errors[-1]}", flush=True)
    issues = errors + audit()
    if issues:
        print("\nFace pilot is NOT ready:", flush=True)
        for issue in issues:
            print(f"  - {issue}", flush=True)
        if args.check_only:
            print("Run on the setup node: PYTHONPATH=src uv run --group faces python -P -m mfdime.setup_faces_pilot", flush=True)
        raise SystemExit(1)
    print("Face pilot prerequisites verified. Submit: sbatch jobs/h2_faces_pilot.sbatch", flush=True)


if __name__ == "__main__":
    sys.exit(main())
