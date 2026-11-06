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
# Setup as in the mfdime.h2_pilot docstring. Resume one group: sbatch --array=1 jobs/h2_full.sbatch (same OUT_ROOT,
# unchanged code). Another shared root: sbatch --export=ALL,OUT_ROOT="$PWD/results/h2_full_v2" jobs/h2_full.sbatch.
# Arms jump1/jump4 x shortcut/exact on pairs 340->339, 293->286 and 285->283, seeds 0-4, paired by photo and seed,
# with DiME's settings (60 steps, gain 1, scales 8/10/15 with retries) and TF32; gradients are not normalised and
# update norms are not matched. Cohort: after a pair's first 16 guide-correct photos (practice), every guide-correct
# photo the ViT-B/16 evaluator also calls the source; the first task publishes it atomically and every task adopts it.
# The root records the code identity and settings and refuses changed ones, so the sources mfdime.h2 hashes must stay
# byte-identical while the experiment may still resume. Finished samples are skipped on resume, interrupted ones are
# archived and retried, numerical failures count as unsuccessful, and OOM or missing assets stop the job.
# Layout: results/h2_full_v1/{protocol.json, cohorts/<pair>.json, jobs/group<g>/{env,complete}.json,
#   runs/<pair>/seed<k>/<peek>/<mode>/{config.json, manifest.jsonl, complete.json, grid_00.jpg, samples/<hash>/}}.
# Score with mfdime.h2_score.
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
