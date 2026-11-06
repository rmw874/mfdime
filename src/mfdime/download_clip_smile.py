"""Download only the pinned OpenCLIP checkpoint/config for the CelebA-HQ smile pilot (about 1.7 GB).

Run on the cluster setup/login node: PYTHONPATH=src uv run --group faces python -P -m mfdime.download_clip_smile
The existing FiRe, STEEX, CelebA-HQ and VGG assets are reused from the face experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from mfdime import clip_smile


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=clip_smile.DIRECTORY)
    args = p.parse_args(argv)
    snapshot_download(repo_id=clip_smile.REPO, revision=clip_smile.REVISION, local_dir=args.out_dir,
                      allow_patterns=[clip_smile.WEIGHT_FILE, "open_clip_config.json"])
    for name in (clip_smile.WEIGHT_FILE, "open_clip_config.json"):
        if not (args.out_dir / name).is_file():
            raise SystemExit(f"STOP: incomplete download, missing {name}")
    (args.out_dir / "download.json").write_text(json.dumps({"repo": clip_smile.REPO, "revision": clip_smile.REVISION}, indent=2))
    print(f"OpenCLIP assets ready: {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
