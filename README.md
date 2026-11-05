# mfdime

DiME explains an image classifier with counterfactuals: it edits a photo step by step until the classifier changes its answer. At every step it looks ahead at the finished image by walking the generator through all the remaining steps (with fresh noise at each; the noise-free walk takes the same steps without it), takes the classifier's gradient on that look-ahead and applies it straight to the noisy image (DiME's shortcut) instead of differentiating through the look-ahead (the exact gradient). We replace the walk with one jump of a flow map: DMF on ImageNet and FiRe's pixel MeanFlow on CelebA-HQ faces.

## Hypotheses

### H1 
With DiME's shortcut, one jump gives a near-identical flip rate and sFID (ACE) as the noise-free walk over 5 seeds, at 3055/212 approx= 14x fewer calls to the generator.

### H2
With the same DiME settings and flow-map look-ahead, full backpropagation through the look-ahead improves target-class
success on a classifier that never guided the edits, compared with DiME's shortcut gradient.

The comparison uses the same gains and sampling schedule, not equal realised update norms: gradient magnitude and
direction are both allowed to change. Independent-classifier success measures transfer, a proxy for semantic validity,
not a direct measurement of visual realism.

### H3 
The shortcut points away from the exact gradient most at high noise and carries more high-frequency energy. The jump's exact gradient points the same way as the walk's.

## H2 implementation design

The four-arm pilot, held-out runner (`mfdime.h2`), and paired bootstrap scorer (`mfdime.h2_score`) are implemented.

### Experiment

- Start with ImageNet and the existing three tasks: zebra -> sorrel (340 -> 339), cheetah -> cougar (293 -> 286),
  and Egyptian cat -> Persian cat (285 -> 283).
- Use the existing SiT/DMF checkpoints and SD-VAE, ResNet-50 guide, and pretrained ViT-B/16 independent evaluator.
  Keep the evaluator checkpoint and preprocessing fixed across arms; never use its predictions for guidance,
  scale retries, or choosing gains.
- Run four arms: jump1/shortcut, jump1/exact, jump4/shortcut, jump4/exact. The primary comparisons are exact versus
  shortcut within each jump count. Jump4 means four composed jumps for each look-ahead, not four guidance updates.
- Preserve DiME's 60 guided steps, gain 1.0, scales 8/10/15, retry logic, L1/perceptual weights, and posterior sampling.
  Do not normalise gradients, match update norms, or tune gains separately by arm.
- Pair photos and seeds 0-4 across all arms. Starting noise and per-scale sampling-noise streams depend only on photo,
  seed, and scale, never on gradient mode or jump count. Arms can take different retries and visit different states.

### Gradient implementation

- Keep shortcut as the default and preserve its existing behaviour, including H1 runs.
- Add an explicit gradient-mode setting, recorded in configuration and every manifest row.
- In exact mode, differentiate the classifier and perceptual losses through decoding and every jump to the current
  noisy state. The generator and decoder weights remain frozen; gradients with respect to their inputs remain enabled.
- Replace the shortcut's approximate look-ahead derivative with the actual derivative, retaining the sampler's
  coordinate conversions. Do not apply the shortcut's inverse-signal derivative approximation on top of the full gradient.
- Retain the direct L1 term. Keep the first guided step identical in both modes: it uses the original image without
  a look-ahead, as the existing implementation does.
- Preserve the existing straight-through output clamp. Here "exact" means backpropagation through the implemented
  look-ahead and loss, including that surrogate derivative.
- Differentiate through all four jumps for jump4. Release the graph after each guidance update; do not backpropagate
  through the history of the outer sampling loop or request higher-order gradients.
- Begin with FP32 and batch size one for GPU pilots. Measure memory before increasing the batch size; add gradient
  checkpointing only if required.

### Evaluation and reporting

- Preserve the current practice/held-out boundary, defined using guide-correct source photos. Within each split,
  use originals that both guide and evaluator classify as the source for the primary H2 cohort. Evaluator filtering
  must not move practice images into held-out data. Record eligible counts and exact photo identities before generation.
- Primary metric: evaluator target-class success over all eligible originals, including guide failures.
- Secondary metrics: guide target success, joint guide/evaluator target success, evaluator success among guide
  successes, pixel L1 change, the existing perceptual distance, retry-scale usage, runtime, and peak GPU memory.
  Record evaluator predictions before and after editing. The full H2 scorer does not compute FID/sFID.
- Report exact-minus-shortcut percentage-point differences for each jump count, pooled across eligible photos and
  separately by class pair. Average over seeds and bootstrap photos within each class pair, preserving all arms and
  seeds together. Use 95% intervals for estimation and report both comparisons regardless of outcome.
- Count the held-out cohort before final runs. The existing H1 pool has 73 photos before evaluator filtering;
  five seeds do not replace independent photos. Any expansion of the class-pair list must be fixed before inspecting
  held-out H2 outcomes. Final sample size remains to be decided after the cohort audit.

### Verification, pilot, and cluster setup

1. Implement the modes, output records, H2 scorer, and Slurm scripts locally. Verify full gradients against a known
   differentiable toy model and finite differences, verify jump4 composition, preserve shortcut behaviour, check noise
   pairing and batch-independent results within numerical tolerances, and exercise all four arms with synthetic data.
   Check that the scorer keeps gradient modes distinct and handles paired cohorts correctly.
2. Rsync the code when ready. On the cluster's internet-enabled setup/login node, initialise pinned submodules,
   install the locked Python environment, and obtain the ImageNet assets using the repositories, revisions, and paths
   in src/mfdime/config.py. ImageNet requires accepting access terms and authenticating. Pre-cache the ResNet, ViT,
   and VGG weights (and Inception if scoring FID). Compute jobs should run from local assets without downloads.
3. Pilot all four arms on the same four practice photos per class pair and one seed, using batch size one and the
   production 60-step schedule. Record losses, finite gradients, update norms, peak memory, timing, predictions, and
   image grids. Fewer than four eligible practice photos in a pair should be reported explicitly.
4. Verify the pilot on the available 32 GB GPU allocation, then freeze the protocol, final photo list, batch size,
   and job resource requests. Pilot outcomes establish correctness and feasibility, not the preferred gradient arm.
5. Submit exactly three single-GPU held-out jobs: seeds 0/1, seeds 2/3, and seed 4. Each job loads the models once
   and processes all three pairs and all four arms sequentially, at batch size one. Checkpoint each sample and
   score only complete, matched outputs. The full job script requests six hours.

No cluster connection or model downloads are required for synthetic verification. Pilots measured a maximum of
8.83 GiB allocated GPU memory at batch size one. Production runtime depends on cohort size and DiME's retries.

## Running the H2 pilot on Slurm

The pilot runs all four arms sequentially for each class pair, with three class pairs in parallel. Each arm edits
the same first four eligible practice photos with seed 0, batch size 1, and the full 60-step DiME schedule. If fewer
than four practice photos pass both classifiers, it uses all eligible photos and prints a warning; zero eligible
photos stops that task. At most 48 counterfactuals are generated across the three tasks. The evaluator never guides
the edits or decides retries. No gradient normalisation is applied.

On your local machine, copy the code to your chosen cluster directory (replace the host and destination):

```bash
rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='runs' \
  --exclude='results' --exclude='weights' --exclude='data' \
  ./ YOUR_CLUSTER:~/mfdime/
```

On the cluster, from the repository root, prepare the environment and downloads on an internet-enabled setup/login
node, following the cluster's policy for large downloads. The pinned input repositories, revisions and destinations
are in `src/mfdime/config.py`; the helper uses those settings. ImageNet access must first be accepted at
[the dataset page](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

```bash
cd ~/mfdime
git submodule update --init third_party/dmf
uv sync --frozen
uv run hf auth login
PYTHONPATH=src uv run python -P -m mfdime.download_h2
make lint typecheck test
mkdir -p logs
sbatch jobs/h2_pilot.sbatch
```

The downloader fetches DMF, SiT, SD-VAE and all 14 ImageNet validation shards, and caches the pretrained ResNet-50,
ViT-B/16 and VGG19 weights. It does not fetch the face assets or Inception, which the H2 pilot does not use. Downloads
can be rerun to reuse completed files. The rsync command includes Git metadata needed to initialise the submodule;
if you copy code another way, retain that metadata or use a proper clone.

The Slurm script requests one A100, 32 GB of host RAM and two hours per task; `--mem=32G` is host RAM, not a GPU-memory
limit. It logs the actual GPU memory available. Adjust partition/GPU resource syntax for your cluster if needed.
To try one class pair before launching all three, submit `sbatch --array=0 jobs/h2_pilot.sbatch`, then submit
`sbatch --array=1-2 jobs/h2_pilot.sbatch` for the remaining pairs.

Results appear under `results/h2_pilot_v1/<source>_<target>/`:

- `photos.json`: original practice pool, evaluator predictions, eligible count and selected identities.
- `env.json`: environment, GPU information and submodule revisions.
- `jump1/shortcut/`, `jump1/exact/`, `jump4/shortcut/`, `jump4/exact/`: configuration, manifests, original/edited
  PNGs, `grid_00.jpg`, and per-step `diagnostics.jsonl` with losses and per-image gradient/update norms.
- `pilot.md` and `pilot.json`: descriptive success rates, paired differences, image distances, scale usage,
  elapsed times and peak allocated GPU memory. Timing includes sampling, final decoding and diagnostic logging,
  but excludes independent evaluation and image-distance scoring. Predictions are made on the in-memory images
  before PNG quantisation. Memory peaks cover sampling and final decoding with all models resident.
- `complete.json`: written only after all four arms and their reports succeed. Partial folders without this marker
  are not completed pilots.

Existing destinations are never overwritten. For a fresh rerun:

```bash
sbatch --export=ALL,OUT_ROOT="$PWD/results/h2_pilot_v2" jobs/h2_pilot.sbatch
```

Review image grids, finite diagnostics, update magnitudes and memory/runtime before fixing the production batch
size and final cohort. Pilot success rates on these practice photos are not a held-out H2 result. The old
`mfdime.score` command remains H1-only and rejects exact-gradient manifests.

For local synthetic verification without pretrained weights or submodule source, run `make test` in a compatible
Python environment. The tests exercise full gradients against finite differences, composition through four jumps,
the unchanged shortcut, shared noise across batch sizes, practice-cohort selection and complete four-arm output.

## Full held-out H2 experiment

The full experiment uses the same four arms and sampling settings as the pilot, including TF32, raw gradients,
gain 1, scale retries, and batch size one. No extra determinism controls or magnitude matching are introduced.
The three Slurm array tasks divide the seeds as follows (definitions are in `src/mfdime/config.py`):

| Array task | Seeds | Work on its one GPU |
|---|---|---|
| 0 | 0, 1 | All three pairs, all four arms |
| 1 | 2, 3 | All three pairs, all four arms |
| 2 | 4 | All three pairs, all four arms |

Within each job, one process loads the models once and processes samples sequentially. Task 2 has half the seed
work of tasks 0 and 1 and should normally finish earlier. Jobs request one A100, 32 GB host RAM and six hours each.

There is no separate preparation job. The first task to finish selecting a pair's held-out cohort publishes its
complete cohort record atomically; every task adopts that same record. Selection first reserves the first 16
guide-correct source images for practice, then filters the remaining guide-correct images by original-image ViT
correctness. All eligible held-out photos are retained, with no cap or outcome-based exclusions. Later jobs and
resubmissions load the frozen identities without re-filtering them. The cohort files report the actual sample
count; the H1 runs had 73 candidates in total before the independent-evaluator filter.

After copying the updated code to the cluster, use the already downloaded pilot assets. From the repository root:

```bash
make lint typecheck test
mkdir -p logs
sbatch jobs/h2_full.sbatch
```

All three tasks use `results/h2_full_v1/`. To select another shared destination:

```bash
sbatch --export=ALL,OUT_ROOT="$PWD/results/h2_full_v2" jobs/h2_full.sbatch
```

Do not give each task a separate output root: the common root freezes the experiment and cohorts. The experiment
records the generation code identity, model/data paths, and protocol settings; it refuses to mix changed code or
settings into an existing root. Identical settings and seeds do not guarantee identical numerical replay across
jobs; this remains the chosen execution policy.

If a task is interrupted or times out, resubmit only its array index, with the same output root and unchanged code:

```bash
# Example: resume the job responsible for seeds 2 and 3.
sbatch --array=1 jobs/h2_full.sbatch
```

Each finished photo/seed/arm has an atomic completion record and is skipped on resume. An incomplete sample's
directory is preserved under its arm's `interrupted/` directory before the same sample is retried. A file lock
prevents two simultaneous copies of the same seed group. Missing assets, OOM, and other infrastructure errors
stop the job rather than changing the experiment. Nonfinite sampling/loss failures are explicitly recorded as
unsuccessful outcomes in the original denominator; they are not retried until a favourable result is obtained.
Finite but visually corrupted edits remain ordinary outcomes and are never discarded.

The output layout is:

```text
results/h2_full_v1/
  protocol.json
  cohorts/<source>_<target>.json
  jobs/group<0|1|2>/{env.json,complete.json}
  runs/<source>_<target>/seed<0..4>/<jump1|jump4>/<shortcut|exact>/
    config.json
    manifest.jsonl
    complete.json
    grid_00.jpg
    samples/<photo_identity_hash>/
      identity.json
      manifest.jsonl
      diagnostics.jsonl
      complete.json
      images/...
```

Grids contain only samples with final images; numerical failures are listed in the manifests and still count in
success-rate denominators. A group completion marker is written only after all its assigned work finishes.

Once all three jobs finish, run the scorer on the cluster or after rsyncing the complete results folder locally:

```bash
PYTHONPATH=src uv run python -P -m mfdime.h2_score \
  --runs results/h2_full_v1 --out-dir results/h2_full_v1/score
```

The scorer validates group, arm, sample and cohort identities and rejects missing/partial data. It writes
`score.json` and `score.md` and refuses to overwrite an existing score directory. The primary endpoint is ViT's
target success over every eligible original, including guide failures. Both exact-minus-shortcut comparisons
are reported separately with 95% paired bootstrap intervals from 5,000 draws. Photos are sampled within each
class pair with all their seeds and arms kept together; pooled results weight each photo equally. Pair-specific
results and secondary guide/joint success, conditional transfer, image distances, scale usage, timing and memory
are also reported. Continuous metrics include the number of available observations because numerical failures
have no final-image distance. There is no automatic overall H2-support label.

To copy back results and logs (run locally, replacing the cluster hostname/path):

```bash
rsync -av YOUR_CLUSTER:~/mfdime/results/h2_full_v1/ results/h2_full_v1/
rsync -av YOUR_CLUSTER:~/mfdime/logs/ logs/
```

Synthetic tests additionally verify the seed partition, concurrent cohort publication, practice/held-out
separation, sample-level resume and interrupted-attempt preservation, numerical-failure accounting, and that
repeating seeds does not artificially narrow the photo-level bootstrap intervals.

## Simple CelebA-HQ / OpenCLIP smile pilot

This is a separate practice-only pilot, run in one job on one GPU:

1. Check frozen OpenCLIP smile predictions against the annotations on all 16 existing practice faces, before
   filtering. Report accuracy, smile/non-smile recall, balanced accuracy, confusion matrix and per-photo scores.
2. Select up to four originals that STEEX and OpenCLIP both label correctly, preferring two smiling and two
   non-smiling faces. The target is the opposite smile label; no generated outputs influence selection.
3. Run `jump1/shortcut`, `jump1/exact`, `jump4/shortcut`, and `jump4/exact` with seed 0, batch 1, the full 60-step
   schedule, existing face gain 3 and DiME retries. OpenCLIP never guides the edits or chooses scales.

The evaluator is `ViT-L-14`, LAION-2B checkpoint `laion2b_s32b_b82k`, at revision
`1627032197142fbe2a7cfec626f4ced3ae60d07a`. The two fixed text prompts are:

- Class 0: `a photo of a person with a neutral facial expression.`
- Class 1: `a photo of a person smiling.`

Choose the class with larger image/text cosine similarity; these scores are not calibrated probabilities.
Both originals and edits are converted to uint8 RGB and use the checkpoint's matching evaluation transform
(224 px, bicubic resize/centre crop, mean/std 0.5). The neutral prompt is an operational proxy for the dataset's
non-smiling class, so the unfiltered original-image check is essential. This small pilot does not establish
reliable smile-recognition performance on the full dataset.

After rsyncing the updated code, run on the cluster setup/login node:

```bash
uv sync --frozen --group faces
PYTHONPATH=src uv run --group faces python -P -m mfdime.setup_faces_pilot
make faces-check
mkdir -p logs
sbatch jobs/h2_faces_pilot.sbatch
```

The complete setup helper reuses existing files and downloads every missing pilot asset to its configured path:

- The pinned FiRe submodule and pMF smile checkpoint.
- STEEX's official `v.0.1/checkpoints_decision_densenet.tar.gz` release archive (230 MB). Only its CelebAMask-HQ
  decision checkpoint is copied to `weights/steex/decision_densenet/celebamaskhq/checkpoint.tar`.
- The pinned CelebAMask-HQ zip and CelebA evaluation partition file.
- Cached VGG19 perceptual weights.
- OpenCLIP's pinned weights/config (about 1.7 GB) under `weights/openclip/ViT-L-14-laion2b/`.

It reports setup/download errors together and exits unsuccessfully until prerequisites pass. If a Hugging Face
source requires authentication, use `uv run --group faces hf auth login` on the setup node and rerun setup.
Dataset terms still apply. The source revisions and paths match `src/mfdime/config.py`; STEEX's release asset
and the pinned dataset filenames have been checked against their official metadata.

To check everything offline without downloading, before allocating a GPU:

```bash
PYTHONPATH=src uv run --group faces python -P -m mfdime.setup_faces_pilot --check-only
```

The check reports all missing files in one pass, verifies the FiRe revision and OpenCLIP download metadata,
and checks the CelebA-HQ test-partition mapping and practice-image entries in the zip. The Slurm job runs this
same offline check before loading the models. Submit only after setup prints `Face pilot prerequisites verified`.
Compute jobs use local assets without network access. The optional `faces` dependency group adds OpenCLIP
without upgrading existing locked dependencies; the ImageNet generation source files and resume identities
are unchanged. The earlier `mfdime.download_clip_smile` helper remains available for OpenCLIP-only downloads.

Results go to `results/h2_faces_pilot_v1/`:

- `evaluator.md`, `evaluator.json`, `originals.csv`, `originals.jpg`: the unfiltered 16-original smile check.
- `photos.json`: selected identities, labels, targets and indices into the practice pool.
- Each arm's directory: original/edited images, `grid_00.jpg`, manifest, configuration and step diagnostics.
  Manifests include OpenCLIP predictions, before/after smile cosine margins, target-directed margin change,
  pixel/perceptual distances, timing and memory.
- `pilot.md` and `pilot.json`: four-arm success summaries; JSON also separates adding and removing smiles.
- `complete.json`: written after all four arms and their reports finish.

If no faces pass the original-label checks, the evaluator report is still saved and the job stops explicitly.
If fewer than four pass or only one smile class is represented, it reports that limitation and runs the
available faces. Existing output directories are never overwritten. For a fresh rerun, use:

```bash
sbatch --export=ALL,OUT_ROOT="$PWD/results/h2_faces_pilot_v2" jobs/h2_faces_pilot.sbatch
```

The synthetic tests run without downloading weights; the optional OpenCLIP integration test uses its real API
with a tiny random model. Install the faces group to include that test. Return this pilot's whole results
folder and `.out`/`.err` logs for review before extending it to held-out faces.

## Full held-out faces experiment

Use `jobs/h2_faces_full.sbatch` for the full CelebA-HQ/OpenCLIP experiment. It submits five array tasks, seeds 0-4,
with one A100 per task. Each job loads the models once and runs all four arms sequentially at **batch size one**,
using the existing pilot sampler. There is no batching benchmark or new sampler implementation.

All tasks share `results/h2_faces_full_v1/`. The first task to publish a complete cohort fixes the same 512 faces
for every seed and arm. Selection follows CelebA-HQ index order, excludes the 16 practice faces and the next 64
faces reserved for prior gain/tuning checks, and retains originals where STEEX and OpenCLIP both match the
annotated smile label. The target is the opposite label. If fewer than 512 eligible faces exist, the job stops
explicitly without silently reducing the sample. Original-image selection never sees generated outcomes.

The checkpoint, preprocessing, fixed prompts, gain 3, 60-step schedule, scale retries and raw gradients match the
pilot. No extra reproducibility controls are introduced. After rsyncing the updated code, run on the cluster from
the repository root:

```bash
uv sync --frozen --group faces
PYTHONPATH=src uv run --group faces python -P -m mfdime.setup_faces_pilot --check-only
make faces-check
mkdir -p logs
sbatch jobs/h2_faces_full.sbatch
```

The pilot assets are sufficient; no new weights or datasets are required. If the offline check reports anything
missing, rerun `mfdime.setup_faces_pilot` without `--check-only` on the setup node before submitting.

Each job requests six hours. The practice pilot suggests approximately 5.7 hours of generation per seed across
the four arms, before setup, scoring and file I/O; this is only an estimate and some jobs may time out. Every
completed image has its own checkpoint and is reused on resubmission. For example, resume seed 2 with:

```bash
sbatch --array=2 jobs/h2_faces_full.sbatch
```

Use unchanged code/settings and the same shared output root when resuming. To start a separate experiment root:

```bash
sbatch --export=ALL,FACES_ROOT="$PWD/results/h2_faces_full_v2" jobs/h2_faces_full.sbatch
```

Do not use a different root for each seed. Completion records prevent mixing different protocols and preserve
per-image results, including numerical failures. Interrupted sample directories are archived before retry;
numerical generator failures count as unsuccessful outcomes and are not rerun to obtain a favourable image.
OOM, evaluator/measurement errors and missing assets stop the job rather than being scored as generator failures.
Finite corrupted outputs remain in the analysis.

Output includes `protocol.json`, `cohort.json`, and `jobs/seed<k>/{env.json,complete.json}`. Each arm lives under
`runs/seed<k>/<jump1|jump4>/<shortcut|exact>/`, with a manifest, configuration, comparison grids and completion record.
Its `samples/<photo_hash>/` folders contain original/edited images, per-step diagnostics and individual completion
records. Grids omit numerical failures, which remain explicitly represented in manifests and success denominators.

Once all five seeds finish, run the dedicated face scorer:

```bash
PYTHONPATH=src uv run --group faces python -P -m mfdime.h2_faces_score \
  --runs results/h2_faces_full_v1 --out-dir results/h2_faces_full_v1/score
```

The scorer verifies the frozen cohort and complete paired coverage, reports pooled and separate adding/removing
smile results, and bootstraps photos within those two directions while retaining all five seeds and all arms
together. It uses 5,000 draws and reports separate 95% intervals for each exact-minus-shortcut comparison. Pooled
results weight faces equally. Secondary metrics include guide/joint success, conditional transfer, pixel/VGG
distances, target-directed OpenCLIP cosine-margin changes, scale usage, generation timing and peak GPU memory.
Cosine margins are not calibrated probabilities. There is no automatic overall H2-support label.

To return the results and logs, run locally:

```bash
rsync -av YOUR_CLUSTER:~/mfdime/results/h2_faces_full_v1/ results/h2_faces_full_v1/
rsync -av YOUR_CLUSTER:~/mfdime/logs/ logs/
```
