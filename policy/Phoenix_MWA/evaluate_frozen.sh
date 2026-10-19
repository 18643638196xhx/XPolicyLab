#!/usr/bin/env bash
set -euo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS="$(cd "${1:?Pass the downloaded weight directory}" && pwd)"; shift
export ROBOTWIN_CHECKPOINT="$WEIGHTS/policy.pt"
export DIFFSYNTH_MODEL_BASE_PATH="$WEIGHTS"
export DINO_ENCODER_CHECKPOINT="$WEIGHTS/encoders/dino110k.pt"
export STVAE_CHECKPOINT="$WEIGHTS/encoders/stvae150k.pt"
export ROBOTWIN_INFERENCE_CONFIG="$P/source/configs/inference/phoenix_robotwin.yaml"
export ROBOTWIN_EVAL_STATS="$P/source/assets/stats/robotwin.json"
exec bash "$P/source/scripts/eval_robotwin.sh" "$@"
