#!/bin/bash
#SBATCH --job-name=mfdime-h2-faces-pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --no-requeue
# One GPU, four faces, seed 0, all four jump/gradient arms. Run mfdime.setup_faces_pilot on the setup node first.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
readonly BASE="$PWD" PYTHON="$PWD/.venv/bin/python"
readonly OUT="${OUT_ROOT:-$BASE/results/h2_faces_pilot_v1}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$BASE/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"$PYTHON" -P -m mfdime.setup_faces_pilot --check-only
"$PYTHON" -c "import open_clip, torch; assert torch.cuda.is_available(), 'STOP: no CUDA'; print(torch.__version__, torch.cuda.get_device_name(0))"
"$PYTHON" -P -m mfdime.h2_faces_pilot --out-dir "$OUT"
