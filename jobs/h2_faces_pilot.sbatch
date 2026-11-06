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
# Setup: uv sync --frozen --group faces; PYTHONPATH=src uv run --group faces python -P -m mfdime.setup_faces_pilot;
# make faces-check. A fresh rerun: sbatch --export=ALL,OUT_ROOT="$PWD/results/h2_faces_pilot_v2" <this script>.
# The pilot checks OpenCLIP against the labels on all 16 practice faces (accuracy, per-class recall, balanced
# accuracy, confusion matrix, per-photo scores), picks up to four faces that STEEX and OpenCLIP both label correctly
# (two smiling and two not where possible; the target is the opposite label) and runs the four arms with gain 3 and
# DiME's retries; OpenCLIP never guides the edits or chooses scales. With no eligible face the evaluator report is
# saved and the job stops; with fewer than four, or one smile class only, it says so and runs what it has.
# Output (never overwritten): evaluator.json/.md, originals.csv/.jpg, photos.json, one directory per arm (images,
# grid, configuration, diagnostics, and a manifest with OpenCLIP predictions, smile margins, distances, timing and
# memory), pilot.json/.md (the JSON split by adding and removing smiles) and complete.json once all arms finished.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
readonly BASE="$PWD" PYTHON="$PWD/.venv/bin/python"
readonly OUT="${OUT_ROOT:-$BASE/results/h2_faces_pilot_v1}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$BASE/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"$PYTHON" -P -m mfdime.setup_faces_pilot --check-only
"$PYTHON" -c "import open_clip, torch; assert torch.cuda.is_available(), 'STOP: no CUDA'; print(torch.__version__, torch.cuda.get_device_name(0))"
"$PYTHON" -P -m mfdime.h2_faces_pilot --out-dir "$OUT"
