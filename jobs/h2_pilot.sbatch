#!/bin/bash
#SBATCH --job-name=mfdime-h2-pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --no-requeue
# Submit from the repo root after the login-node setup documented in README.md.
# Each GPU task runs all four arms for one pair, four practice photos, seed 0, batch 1.
# OUT_ROOT may select a fresh destination for reruns; existing results are never overwritten.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
readonly BASE="$PWD" PYTHON="$PWD/.venv/bin/python"
readonly SOURCES=(340 293 285) TARGETS=(339 286 283)
readonly SRC="${SOURCES[$SLURM_ARRAY_TASK_ID]}" TGT="${TARGETS[$SLURM_ARRAY_TASK_ID]}"
readonly OUT="${OUT_ROOT:-$BASE/results/h2_pilot_v1}/${SRC}_${TGT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$BASE/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
[[ ! -e "$OUT" ]] || { echo "STOP: $OUT already exists; choose a fresh OUT_ROOT" >&2; exit 1; }
readonly TORCH_CACHE="${TORCH_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/torch}/hub/checkpoints"
for f in third_party/dmf/models/dmft.py weights/dmf/dmf_xl_2_256.pt weights/dmf/sit_xl_2_repa_256.pt \
         weights/sd-vae-ft-ema/config.json "$TORCH_CACHE/resnet50-0676ba61.pth" \
         "$TORCH_CACHE/vit_b_16-c867db91.pth" "$TORCH_CACHE/vgg19-dcbb9e9d.pth"; do
    [[ -e "$f" ]] || { echo "STOP: missing $f; complete login-node setup" >&2; exit 1; }
done
shopt -s nullglob
shards=(data/imagenet-1k/data/validation-*.parquet)
[[ ${#shards[@]} -eq 14 ]] || { echo "STOP: found ${#shards[@]} of 14 validation shards" >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'STOP: no CUDA'; print(torch.__version__, torch.cuda.get_device_name(0))"
"$PYTHON" -P -m mfdime.h2_pilot --source "$SRC" --target "$TGT" --n 4 --seed 0 --batch 1 --out-dir "$OUT"
