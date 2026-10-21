## Policy

Add Phoenix-MWA's RoboTwin 2.0 posttraining, evaluation and checkpoint integration under `policy/Phoenix_MWA`. The bridge exposes the released policy through XPolicyLab while retaining the source RGB processing, z-score normalization, action mapping and 32-step replanning.

- Name: [Phoenix-MWA](https://openphoenix-team.github.io/Phoenix-pages/).
- Upstream: [OpenPhoenix-Team/Phoenix-MWA, post-train](https://github.com/OpenPhoenix-Team/Phoenix-MWA/tree/post-train), pinned to `16bc6e307ee71b3797e03a4f072a69340f078dab`.
- Supported: `bench_name=RoboTwin`, `env_cfg_type=arx_x5`, `action_type=joint`; the simulator uses aloha-agilex.
- Training support: full posttraining integration, with explicit initialization, dataset and text-cache paths.

## Components

- [x] `install.sh`
- [x] `model.py` and `__init__.py`
- [x] Images: server-decoded RGB, no adapter channel swaps or custom XPolicyLab image-bit decoder. Training consumes the documented native LeRobot v2.1 video export.
- [x] `deploy.yml` with the standard keys and matching policy name
- [x] `deploy.py` aligned with `demo_policy`
- [x] `eval.sh`, `setup_eval_policy_server.sh`, `setup_eval_env_client.sh`
- [x] `process_data.sh` and `train.sh`
- [x] README covering installation, data, training, checkpoint download and official evaluation

## Testing

- [x] Shell syntax and Python compilation
- [x] Decode/encode and RGB-path audit
- [x] Nine CPU regression tests, including the real 35K checkpoint contract and source preprocessing
- [x] Complete SHA256 verification of the five downloaded model files, using existing server files
- [x] Training configuration dry-run with real dataset/cache resources; source wheel build
- [ ] XPolicyLab debug closed loop, plain and encoded observations — not yet verified; not covered by the supplied source logs
- [x] Source-protocol simulator evaluation evidence supplied: C2C 4327/5000 (86.54%), C2R 3621/5000 (72.42%); 100 client-log hashes and aggregate counts verified

See `VALIDATION.md` for the scope of each check. The training dry-run used an available 256013-step initialization to validate launching; it does not reproduce the reported 230K-initialized experiment.

## Checkpoint

[AnyverseDynamics/Phoenix-RoboTwin](https://huggingface.co/AnyverseDynamics/Phoenix-RoboTwin): policy with 230000-step initialization and 35000 posttraining steps. The downloader also fetches the public encoders, WanVAE, T5 and tokenizer, with release checksums and pinned HF revisions.

```bash
cd policy/Phoenix_MWA
bash download_checkpoint.sh checkpoints/phoenix-robotwin-35k
```

No weight or dataset bytes are included in the PR.

## Limitations / notes

- A separate 230K initialization is not published in the referenced RoboTwin weight repository. Training requires an explicit compatible initialization; the downloader supports a separately published HF file.
- The upstream native export has specific camera keys and ordered clean-episode blocks. It is not interchangeable with an arbitrary output from the official XPolicyLab converters. The README documents the required layout.
- Current official XPolicyLab evaluation and the source's older frozen-scene protocol have separate instructions. The supplied source results, 86.54% C2C and 72.42% C2R, are archived with [100 task logs and result summaries](https://github.com/ssssxt/XPolicyLab/tree/codex/phoenix-mwa-robotwin2/policy/Phoenix_MWA/evaluation_results/robotwin_20261018). They have not been reproduced by the current integration. Counts come from the supplied structured index; per-episode result files and a model-load hash receipt are absent. Recovery tracebacks are preserved in 13 logs; see the evidence README.
- Vendored common model/data/training code, original licenses and per-file source hashes are retained. The adapter is separate from the Phoenix_MWA RoboDojo policies in PRs #165 and #186.
