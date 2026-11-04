#!/bin/bash
#SBATCH --job-name=mfdime-h1-faces
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-4%5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --no-requeue
# H1 on faces (src/mfdime/explain.py): with DiME's loop as published, does one pMF jump as the peek give the flip rate
# and sFID of the noise-free walk at a fraction of the generator calls? CelebA-HQ smile with pMF and STEEX's
# classifier, one array task per seed (0-4). Every task explains the same 512 held-out faces (--skip 64 leaves out the
# first 64, kept for checking the gain; sFID needs many faces) with two peeks: jump1 and euler_rest (the
# noise-free walk). Expected about 2 h per task on an A100. Needs ~/mfdime/.venv (uv sync --frozen), third_party/fire
# at the pinned SHA, the pMF smile checkpoint, STEEX's classifier, the CelebAMask-HQ zip, CelebA's partition list and
# the VGG19 hub file (src/mfdime/config.py). GAIN is required, the classifier strength chosen on the practice
# faces:
#   sbatch --export=ALL,GAIN=3 jobs/h1_celebahq.sbatch
# Writes results/h1_celebahq_gain<GAIN>_v1/seed<k>/<peek>/ (each refuses to overwrite).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
readonly BASE="$PWD"
[[ -n "${GAIN:-}" ]] || { echo "STOP: set GAIN (sbatch --export=ALL,GAIN=...)" >&2; exit 1; }
readonly SEED="$SLURM_ARRAY_TASK_ID" N=512 PEEKS=(jump1 euler_rest)
readonly OUT="$BASE/results/h1_celebahq_gain${GAIN}_v1/seed$SEED" PYTHON="$BASE/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$BASE/src" HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$PYTHON" -c "import sys, torch; assert torch.cuda.is_available(), 'STOP: no CUDA'; print(sys.version, torch.__version__, torch.cuda.get_device_name(0))"
for peek in "${PEEKS[@]}"; do
    [[ ! -e "$OUT/$peek" ]] || { echo "STOP: $OUT/$peek exists (runs are written once)" >&2; exit 1; }
done
readonly HUB="$HOME/.cache/torch/hub/checkpoints"
for f in third_party/fire/src/pmf/dit.py weights/fire/pmf-celebahq256-smile.pt data/celebahq/CelebAMask-HQ.zip \
         data/celebahq/list_eval_partition.txt weights/steex/decision_densenet/celebamaskhq/checkpoint.tar \
         "$HUB/vgg19-dcbb9e9d.pth"; do
    [[ -e "$f" ]] || { echo "STOP: $f missing (login-node setup)" >&2; exit 1; }
done
for peek in "${PEEKS[@]}"; do
    echo "== $peek, $N faces, seed $SEED, $(date -Is)"
    "$PYTHON" -P -m mfdime.explain --dataset celebahq --split held_out --skip 64 --n "$N" \
        --gain "$GAIN" --seeds "$SEED" --peek "$peek" --out-dir "$OUT/$peek"
done
echo "== done $(date -Is)"
