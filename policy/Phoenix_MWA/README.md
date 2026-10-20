# Phoenix-MWA — RoboTwin 2.0

RoboTwin-specific **train / eval / checkpoint** integration of [Phoenix-MWA](https://github.com/OpenPhoenix-Team/Phoenix-MWA/tree/post-train), pinned to source commit `16bc6e307ee71b3797e03a4f072a69340f078dab`. [Project and technical blog](https://openphoenix-team.github.io/Phoenix-pages/). This directory is independent of the Phoenix_MWA RoboDojo adapters in PRs #165 and #186.

Supported: `bench_name=RoboTwin`, `env_cfg_type=arx_x5`, `action_type=joint`. RoboTwin's simulator embodiment is **aloha-agilex**; `arx_x5` is its XPolicyLab action profile, as specified in the [official integration guide](https://robotwin-platform.github.io/doc/usage/xpolicylab.html). This does not declare real-robot RoboDojo support.

The released [policy checkpoint](https://huggingface.co/AnyverseDynamics/Phoenix-RoboTwin) records **230000-step initialization → 35000-step posttraining**. The adapter preserves three separate RGB views, the source resize/center-crop processor, z-score normalization, canonical80 mapping, BF16, repeat9/full STVAE current conditioning, online T5, 32-action horizon/replanning and 10 denoising steps. These settings differ from the RoboDojo H5 releases.

## Installation

Run from the policy directory inside a RoboTwin workspace containing `XPolicyLab/` and the external `env_cfg/`:

```bash
cd XPolicyLab/policy/Phoenix_MWA
bash install.sh phoenix-mwa
conda activate phoenix-mwa
```

Python 3.10, PyTorch 2.7.1 and torchvision 0.22.1 are pinned by the source package. Training additionally installs Accelerate and DeepSpeed/ZeRO-1. Use a compatible NVIDIA driver/CUDA environment. The policy environment does not install SAPIEN, CuRobo or simulator assets; follow [RoboTwin installation](https://robotwin-platform.github.io/doc/usage/install.html) in a separate simulator environment.

`source/` contains the upstream shared model/data/training library and only the RoboTwin-specific task configs, scripts, statistics and evaluation code. Original licenses and notices are retained. Source hashes and provenance are in `source_manifest.json`. No dataset or weight bytes are committed.

## Checkpoints

```bash
bash download_checkpoint.sh checkpoints/phoenix-robotwin-35k
# Offline verification of a completed download:
bash download_checkpoint.sh checkpoints/phoenix-robotwin-35k --verify-only
```

The downloader uses immutable HF revisions for the policy, encoders and tokenizer; large model files are verified by SHA256 and size. It downloads only the inference dependencies:

```text
checkpoints/phoenix-robotwin-35k/
├── policy.pt
├── encoders/dino110k.pt
├── encoders/stvae150k.pt
├── DiffSynth-Studio/Wan-Series-Converted-Safetensors/
│   ├── Wan2.2_VAE.safetensors
│   └── models_t5_umt5-xxl-enc-bf16.safetensors
└── Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
```

Sources: [policy](https://huggingface.co/AnyverseDynamics/Phoenix-RoboTwin), [DINO/CS-VAE encoders](https://huggingface.co/AnyverseDynamics/Phoenix-Encoders), [WanVAE/T5](https://www.modelscope.cn/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors), [tokenizer](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B). See `weights_manifest.json` for exact filenames, revisions and checksums. Inference does not need a pretraining checkpoint, training data, training T5 cache, or original Wan DiT shards.

For training, download the original Wan DiT backbones as well:

```bash
bash download_checkpoint.sh checkpoints/phoenix-robotwin-35k --training-backbones
# Optional: retrieve a separately published initialization checkpoint.
# Replace INIT_REPO / INIT_FILE / INIT_REVISION with the release you intend to train from.
bash download_checkpoint.sh checkpoints/phoenix-robotwin-35k \
  --init-repo INIT_REPO --init-file INIT_FILE --init-revision INIT_REVISION
```

The referenced RoboTwin repository currently publishes the posttrained `policy.pt`, not a separate 230K initialization file. Training accepts `--init /path/to/initialization.pt` and reports its actual step; it never silently substitutes the 35K policy for the 230K initialization. The optional download is saved as `initialization.pt` with a provenance receipt.

## Data processing

The upstream training reader expects its **LeRobot v2.1** RoboTwin export: 27500 episodes, 50 Hz, `robot_type=aloha`, native 14D `action` / `observation.state`, and `observation.images.cam_high`, `cam_left_wrist`, `cam_right_wrist`. It selects the first 50 episodes of each 550-episode task block: **2500 clean demonstrations across 50 tasks**. Randomized trajectories in the surrounding export are excluded by the reader.

This is the prepared upstream export, **not a claim of compatibility with an arbitrary export from XPolicyLab's official converters**. Its camera names, global task-block ordering and clean selector are required. Keep the original `meta/`, `data/` and `videos/` directories. This adapter does not decode raw XPolicyLab HDF5 image bytes or change stored channel conventions. A custom dataset requires a separately reviewed reader/configuration and new statistics; do not relabel unrelated episodes to bypass these checks.

```bash
# Validate an existing native export and its task-indexed text cache:
bash process_data.sh RoboTwin cotrain arx_x5 joint \
  --data /path/to/robotwin-v21 --text-cache /path/to/robotwin-text

# Or build a NEW indexed T5 cache from meta/tasks.jsonl:
bash process_data.sh RoboTwin cotrain arx_x5 joint \
  --data /path/to/robotwin-v21 --text-cache /path/to/new-robotwin-text \
  --build-text-cache --weights "$PWD/checkpoints/phoenix-robotwin-35k" --device cuda:0
```

The cache builder writes `task_index_sharded_mmap_v1`, with BF16 valid tokens and prompt hashes. The training reader restores zero-padded 128×4096 features and the source all-visible attention mask. No visual feature cache is used; training encodes augmented RGB online. Existing caches are checked and never overwritten.

## Training

The standard six positional arguments are followed by explicit resource options:

```bash
bash train.sh RoboTwin cotrain arx_x5 joint 42 0,1,2,3,4,5,6,7 \
  --init /path/to/phoenix-initialization.pt \
  --weights "$PWD/checkpoints/phoenix-robotwin-35k" \
  --data /path/to/robotwin-v21 --text-cache /path/to/robotwin-text \
  --nnodes 1 --nproc 8 --batch-size 8 --accumulation 1 --max-steps 35000
```

Append `--dry-run` to validate resources, metadata, cache and Hydra configuration and print the launch command without starting training. `--init` must use the public Phoenix checkpoint schema. The source factory also requires the original Wan DiT shards downloaded by `--training-backbones` before it restores initialization weights.

For the upstream **8-node × 8-GPU, batch 8/GPU** topology, run on every node with `--nnodes 8 --nproc 8 --node-rank N --master-addr HOST`; global batch is 512 with accumulation 1. The single-node command above has global batch 64 and is not an exact reproduction of that schedule. LR is 5e-5 → 5e-7 with cosine decay and a 5% warmup. The wrapper defaults to a 35000-update budget; upstream's uncapped recipe uses 50 epochs. AdamW, BF16, ZeRO-1, all 28 prompt slots, online RGB augmentation and loss weights are preserved.

Output: `checkpoints/RoboTwin-cotrain-arx_x5-joint-42/`. Existing run directories are refused. On successful completion, rank 0 exports the latest checkpoint as `policy.pt` and links shared frozen resources, so the same directory can be passed to eval. Shared links refer to your downloaded weight directory; include those resources when moving an exported run. This is posttraining code, not a pretraining implementation.

## Evaluation through XPolicyLab

From this directory:

```bash
bash eval.sh RoboTwin adjust_bottle phoenix-robotwin-35k arx_x5 joint 0 \
  0 0 phoenix-mwa robotwin
```

The standard arguments are benchmark, task, checkpoint, action profile, action type, seed, policy GPU, simulator GPU, policy environment and simulator environment. The policy reads already-decoded RGB arrays from the server. Each environment has its own action queue and reset state. No extra clipping or normalization is applied by the bridge.

For official multi-task evaluation, run from the **current RoboTwin repository root**:

```bash
bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/all_tasks.yml \
  --policy-name Phoenix_MWA --ckpt-name phoenix-robotwin-35k \
  --env-cfg-type arx_x5 --action-type joint \
  --task-config demo_clean --test-num 100 --instruction-type unseen \
  --policy-conda-env phoenix-mwa --eval-env-conda-env robotwin --dry-run

bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/all_tasks.yml \
  --policy-name Phoenix_MWA --ckpt-name phoenix-robotwin-35k \
  --env-cfg-type arx_x5 --action-type joint \
  --task-config demo_randomized --test-num 100 --instruction-type unseen \
  --policy-conda-env phoenix-mwa --eval-env-conda-env robotwin --dry-run
```

Review the GPU pool, concurrency and 50-task list in `env_cfg/eval/all_tasks.yml`, then remove `--dry-run`. The CLI options select 100 trials/task, unseen instructions and the clean/randomized task config. Simulator embodiment and randomization come from `env_cfg/task_config/<task-config>.yml`; `arx_x5` selects the action profile. See [official RoboTwin/XPolicyLab instructions](https://robotwin-platform.github.io/doc/usage/xpolicylab.html).

Debug wiring, with a GPU large enough to load the model:

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboTwin adjust_bottle phoenix-robotwin-35k arx_x5 joint 0 0 0 phoenix-mwa robotwin
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboTwin adjust_bottle phoenix-robotwin-35k arx_x5 joint 0 0 0 phoenix-mwa robotwin
```

`deploy.yml` documents 32-step replanning and 10 denoising steps; the source runtime is BF16. Checkpoints resolve with the shared XPolicyLab resolver. For a trained run, pass its concatenated directory name or absolute path. `policy_file` defaults to `policy.pt`; `audit_dir` can select the directory for `MODEL_LOADED.json`.

## Original frozen evaluation protocol

The upstream reported C2C/C2R experiments used a pinned older RoboTwin snapshot, local patches and frozen initial scenes. Those scores must not be described as a fresh run of today's XPolicyLab scheduler. See [ROBOTWIN_ORIGINAL.md](ROBOTWIN_ORIGINAL.md) for the retained source protocol and optional frozen snapshots. Normal official XPolicyLab evaluation does not require those frozen archives.

## Validation and submission

See [VALIDATION.md](VALIDATION.md) for checks actually performed and remaining hardware validation. [PR_DESCRIPTION.md](PR_DESCRIPTION.md) is a submission draft. User-provided results are 86.54% clean2clean and 72.42% clean2random; this integration has not reproduced those success rates. Raw result/log attachments are outside this delivery as requested.
