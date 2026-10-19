# Validation record

Checked on 2026-10-19 against XPolicyLab base `562c14c` and Phoenix-MWA source `16bc6e307ee71b3797e03a4f072a69340f078dab`.

## Completed checks

- Required adapter files and deploy keys are present. `deploy.py` matches `demo_policy/deploy.py`; the evaluation scripts retain its ten positional arguments.
- Python syntax compilation, shell syntax checks, and source snapshot SHA256 checks passed.
- Nine CPU regression tests passed in the existing server Python environment. They exercise action packing, queue acknowledgments, independent batch sessions/reset, invalid observations, RGB view order, repeat9/full-encoder selection, real processor/checkpoint contracts, indexed BF16 text-cache round trips and source integrity.
- The real public policy was inspected with CPU memory mapping: step **35000**, recorded initialization step **230000**, RoboTwin embodiment 2, native joint14 to canonical80, online visual conditioning. A wrong embodiment was rejected by the source contract validator. This checks the checkpoint contract, not a full model forward pass.
- Full SHA256 and size verification passed for the public policy, DINO, STVAE, WanVAE and T5 weights. The tokenizer files were present. These checks used existing server files matching the release manifest; a fresh multi-GB network download was not repeated.
- The source package successfully built a wheel with `pip wheel --no-deps --no-build-isolation`. This checks packaging, not installation into a new environment.
- A training dry-run validated the real native LeRobot v2.1 dataset, all indexed task prompts in its existing text cache, initialization schema, and composed Hydra training recipe, then printed the Accelerate command. The available initialization used for this launch check was **256013**, explicitly reported by the script. It was not the experiment's 230000 initialization and did not run an optimizer step.
- The native export reports 27500 episodes and 50 Hz; the source selector chooses 2500 clean episodes. The existing text cache contains 921032 indexed prompts. The new cache builder's serialization was checked by the CPU round-trip test; a full new GPU cache build was not run.
- Official multi-task command flags were checked against RoboTwin's current `eval_policy_multitask.py`: `--task-config`, `--test-num` and `--instruction-type` are explicit CLI arguments.

CPU regression command, from the parent workspace:

```bash
PHOENIX_ROBOTWIN_TEST_WEIGHT=/path/to/policy.pt \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest XPolicyLab.policy.Phoenix_MWA.test_policy -v
```

## Not run

- Fresh `install.sh` environment installation.
- GPU model loading, inference, or the XPolicyLab debug closed loop with plain and encoded RGB observations.
- A training optimizer step or complete posttraining run; final training export is checked against the upstream manifest schema, not against a newly completed run.
- Current official simulator evaluation, the frozen-source evaluation, or reproduction of the reported success rates.

The inspected server GPUs were already occupied, so no training jobs were interrupted and no GPU validation was started. Before claiming runtime readiness, run both debug commands in the README on a free GPU, then a simulator smoke test. No debug or simulator result is represented as passed here.

## Submission boundaries

This package includes train/eval/checkpoint entry points, not pretraining. The dataset reader requires the documented native export rather than arbitrary XPolicyLab-converted data. The separate 230K initialization is not in the referenced RoboTwin HF repository; use an explicitly selected compatible initialization for training. The public 35K policy and all inference dependencies have download entries.

The 86.54% C2C and 72.42% C2R values were supplied for the original source protocol; this adapter does not establish those results. Logs and result attachments are intentionally outside this delivery.
