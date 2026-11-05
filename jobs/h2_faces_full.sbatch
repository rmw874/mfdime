#!/bin/bash
#SBATCH --job-name=mfdime-h2-faces-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-4%5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --no-requeue
# Five single-GPU jobs, one seed each. Batch one, 512 held-out faces, all four arms sequentially.
# Resubmit an interrupted array index with the SAME shared FACES_ROOT and unchanged code/settings to resume.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
readonly BASE="$PWD" PYTHON="$PWD/.venv/bin/python"
readonly OUT="${FACES_ROOT:-$BASE/results/h2_faces_full_v1}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$BASE/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"$PYTHON" -P -m mfdime.setup_faces_pilot --check-only
"$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'STOP: no CUDA'; print(torch.__version__, torch.cuda.get_device_name(0))"
"$PYTHON" -P -m mfdime.h2_faces --seed "$SLURM_ARRAY_TASK_ID" --out-dir "$OUT"
