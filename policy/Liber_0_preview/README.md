# Liber_0_preview

**Contributor:** LiberAI | **Paper:** Pending | **arXiv:** Pending | **Original code:** Included

RoboDojo evaluation policy for `arx_x5` joint control.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Requires Linux, Python 3.10, a CUDA 13-compatible GPU/driver and FlashAttention 2.8.4. Building FlashAttention may require a matching CUDA toolkit and compiler. Keep the simulator environment separate. This adapter reuses the included `Liber_0_lite` runtime; retain both policy directories. Third-party notices remain in [Liber_0_lite/NOTICE.md](../Liber_0_lite/NOTICE.md).

```bash
cd XPolicyLab/policy/Liber_0_preview
bash install.sh /path/to/env
export PATH=/path/to/env/bin:$PATH
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
python download_checkpoint.py --destination ./checkpoint --assets-dir ./assets
export LIBER0_MODEL_PATH="$PWD/assets"
export LIBER0_CONTEXT_PATH="$PWD/checkpoint/context"
```

Checkpoint: [Liber0-Preview-Robodojo](https://huggingface.co/LiberAI/Liber0-Preview-Robodojo).

The downloader retrieves and verifies the complete checkpoint, including `context/`, at revision `2773a04e3b8f09096ec7beb6105bd6ed80c91da8`. No separate Lite checkpoint download is needed. Base model assets are downloaded separately at a pinned revision.

## Data Processing

Not required for evaluation.

## Training

Evaluation-only. Training code will be released with the paper.

## Evaluation

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls "$PWD/checkpoint" \
  arx_x5 joint 0 0 0 /path/to/env /path/to/env
EVAL_ENV_TYPE=sim bash eval.sh RoboDojo stack_bowls "$PWD/checkpoint" \
  arx_x5 joint 0 0 0 /path/to/env RoboDojo
```

The checkpoint directory contains `model.pt`, `config.yaml` and `dataset_stats.json`. RoboDojo simulator/assets and sibling `env_cfg/` are separate dependencies. Keep `eval_batch: false`; batched evaluation is not supported. Use `DEBUG_OBS_ENCODED=1` to test encoded observations.
