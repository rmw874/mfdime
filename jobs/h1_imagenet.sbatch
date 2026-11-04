#!/bin/bash
#SBATCH --job-name=mfdime-h1-imagenet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --no-requeue
# H1 on ImageNet (src/mfdime/explain.py): with DiME's loop as published, does one DMF jump as the peek give the flip
# rate and sFID of SiT's noise-free walk at a fraction of the generator calls? ResNet-50 guides, at the dataset's gain
# from mfdime.config. One array task per pair of ACE's three (0: zebra -> sorrel 340 -> 339, 1: cheetah -> cougar
# 293 -> 286, 2: Egyptian cat -> Persian cat 285 -> 283), seeds 0-4 in turn: every held-out photo the guide classifies
# right (--n 0) with two peeks, jump1 and euler_rest (the noise-free walk). Expected about 1 h per task on an A100.
# Needs ~/mfdime/.venv (uv sync --frozen), third_party/dmf at the pinned SHA, the DMF/SiT weights, SD-VAE ft-ema, the
# ImageNet validation shards and the torch hub files below (src/mfdime/config.py):
#   sbatch jobs/h1_imagenet.sbatch
# Writes results/h1_imagenet_v1/<source>_<target>/seed<k>/<peek>/ (each refuses to overwrite).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
readonly BASE="$PWD"
readonly SOURCES=(340 293 285) TARGETS=(339 286 283) SEEDS=(0 1 2 3 4) PEEKS=(jump1 euler_rest)
readonly SRC="${SOURCES[$SLURM_ARRAY_TASK_ID]}" TGT="${TARGETS[$SLURM_ARRAY_TASK_ID]}"
readonly OUT="$BASE/results/h1_imagenet_v1/${SRC}_${TGT}" PYTHON="$BASE/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$BASE/src" HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$PYTHON" -c "import sys, torch; assert torch.cuda.is_available(), 'STOP: no CUDA'; print(sys.version, torch.__version__, torch.cuda.get_device_name(0))"
[[ ! -e "$OUT" ]] || { echo "STOP: $OUT exists (runs are written once)" >&2; exit 1; }
readonly HUB="$HOME/.cache/torch/hub/checkpoints"
for f in third_party/dmf/models/dmft.py weights/dmf/dmf_xl_2_256.pt weights/dmf/sit_xl_2_repa_256.pt \
         weights/sd-vae-ft-ema/config.json "$HUB/resnet50-0676ba61.pth" "$HUB/vit_b_16-c867db91.pth" \
         "$HUB/vgg19-dcbb9e9d.pth"; do
    [[ -e "$f" ]] || { echo "STOP: $f missing (login-node setup)" >&2; exit 1; }
done
shopt -s nullglob
shards=(data/imagenet-1k/data/validation-*.parquet)
[[ ${#shards[@]} -eq 14 ]] || { echo "STOP: ${#shards[@]} of 14 ImageNet validation shards (download unfinished?)" >&2; exit 1; }
for seed in "${SEEDS[@]}"; do
    for peek in "${PEEKS[@]}"; do
        echo "== $SRC -> $TGT, $peek, seed $seed, $(date -Is)"
        "$PYTHON" -P -m mfdime.explain --dataset imagenet --source "$SRC" --target "$TGT" --split held_out --n 0 \
            --seeds "$seed" --peek "$peek" --out-dir "$OUT/seed$seed/$peek"
    done
done
echo "== done $(date -Is)"
