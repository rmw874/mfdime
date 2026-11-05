#!/bin/bash
#SBATCH --job-name=mfdime-h2-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --no-requeue
# Exactly three single-GPU jobs. Group 0: seeds 0/1; group 1: seeds 2/3; group 2: seed 4.
# Each loads models once and processes all pairs/arms sequentially, batch one.
# All groups MUST use the same OUT_ROOT. Resubmitting the same array resumes completed samples.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
readonly BASE="$PWD" PYTHON="$PWD/.venv/bin/python"
readonly OUT="${OUT_ROOT:-$BASE/results/h2_full_v1}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$BASE/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
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
"$PYTHON" -P -m mfdime.h2 --group "$SLURM_ARRAY_TASK_ID" --out-dir "$OUT"
