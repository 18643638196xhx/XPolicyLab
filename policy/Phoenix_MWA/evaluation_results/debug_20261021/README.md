# XPolicyLab debug closed-loop validation — 2026-10-21

Both official `eval.sh` debug modes passed on the real, pinned 35K RoboTwin policy (230K initialization). Tested policy source and shared XPolicyLab files match PR commit `097b176cbc11aadd403811a5b8a3112133fbbcf7`; no policy or shared-client/server code was changed for this run. `tested_files.json` records all 475 tested checkout-file hashes (excluding archived evaluation evidence).

| Mode | Completed debug episodes | Exit code | Tracebacks | Final marker |
| --- | ---: | ---: | ---: | --- |
| Plain RGB | 10 | 0 | 0 | `[MAIN] eval finished` |
| Encoded observations (`DEBUG_OBS_ENCODED=1`) | 10 | 0 | 0 | `[MAIN] eval finished` |

Evidence: [plain log](plain.log), [encoded log](encoded.log), [summary](summary.json), [plain model-load receipt](plain_audit/MODEL_LOADED.json), [encoded model-load receipt](encoded_audit/MODEL_LOADED.json), and [exact launcher](run.sh). Debug observations are synthetic; these checks validate the current adapter's client/server loop, not simulator task success.

Both policy and client ran in the existing `fastwam-pretrain` environment, with PyTorch 2.7.1+cu128 and transformers 5.12.1, on GPU 0 (NVIDIA RTX PRO 5000 72GB Blackwell). The launcher uses horizon/replan 32, 10 denoising steps, and the released repeat9/full-STVAE current-view path. A fresh conda installation was not tested.

Before successful runs, the isolated test workspace needed its external `env_cfg/sim/sim_config.yml`; this was copied from the existing evaluation workspace. The old deployment's STVAE link was missing and its policy link pointed to a subsequently migrated checkpoint. The isolated checkpoint directory was relinked to archived original policy/STVAE files. Complete SHA256 and size checks passed for all five model files against the unchanged PR manifest; see `weights_verified.json`. No manifest checks were bypassed and no weights were modified. The two initial setup failures are retained as `plain_attempt1_missing_env.log` and `plain_attempt2_stale_weights.log`.

The successful logs contain a benign shell `libtinfo.so.6` version-information warning; neither successful run contains a Python traceback. Policy-server processes were cleaned up by `eval.sh` after each client completed.
