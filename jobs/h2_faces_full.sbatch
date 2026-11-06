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
# Setup: uv sync --frozen --group faces; PYTHONPATH=src uv run --group faces python -P -m mfdime.setup_faces_pilot
# (--check-only only checks); make faces-check. Resume seed 2: sbatch --array=2 jobs/h2_faces_full.sbatch.
# Another shared root: sbatch --export=ALL,FACES_ROOT="$PWD/results/h2_faces_full_v2" jobs/h2_faces_full.sbatch.
# Cohort: in CelebA-HQ index order, after the 16 practice faces and the next 64 (also left out of H1's face run), the
# first 512 faces where STEEX and OpenCLIP both match the smile label; the target is the opposite label, and fewer
# than 512 eligible faces stop the job. Gain 3, 60 steps, scale retries and raw gradients, as in the pilot. About
# 5.7 h of generation per seed, so a job may time out; finished images are reused. The sources mfdime.h2_faces hashes
# must stay byte-identical while the experiment may still resume. Numerical failures count as unsuccessful and are
# not rerun; OOM, evaluator errors and missing assets stop the job; finite corrupted outputs stay.
# Layout: results/h2_faces_full_v1/{protocol.json, cohort.json, jobs/seed<k>/{env,complete}.json,
#   runs/seed<k>/<peek>/<mode>/{manifest, configuration, grids, completion record, samples/<hash>/}}.
# Score with mfdime.h2_faces_score.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
readonly BASE="$PWD" PYTHON="$PWD/.venv/bin/python"
readonly OUT="${FACES_ROOT:-$BASE/results/h2_faces_full_v1}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$BASE/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"$PYTHON" -P -m mfdime.setup_faces_pilot --check-only
"$PYTHON" -c "import torch; assert torch.cuda.is_available(), 'STOP: no CUDA'; print(torch.__version__, torch.cuda.get_device_name(0))"
"$PYTHON" -P -m mfdime.h2_faces --seed "$SLURM_ARRAY_TASK_ID" --out-dir "$OUT"
