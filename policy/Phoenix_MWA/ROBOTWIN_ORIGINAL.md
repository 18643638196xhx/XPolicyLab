# Retained upstream frozen RoboTwin protocol

This describes the source experiments associated with 86.54% C2C / 72.42% C2R, not a new result from the current XPolicyLab integration. The policy code is shared; simulator protocol and episode inputs differ.

The source pins RoboTwin `13c3c47ff4312dd62484bcd51be034af55c062d1`, CuRobo `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`, and RoboTwin asset revision `9dc9299c163db059931898a9f0852098a61155a1`. The source's simulator/configuration patches and version manifest are retained under `source/assets/robotwin/`. Preserve the upstream notices. See the [pinned setup instructions](https://github.com/OpenPhoenix-Team/Phoenix-MWA/blob/16bc6e307ee71b3797e03a4f072a69340f078dab/docs/ROBOTWIN.md#exact-simulator-and-assets) for installing these resources in a separate new directory. Do not apply these historical patches to the current official simulator used for XPolicyLab submission.

Both frozen archives are publicly listed at [AnyverseDynamics/Phoenix-RoboTwin](https://huggingface.co/AnyverseDynamics/Phoenix-RoboTwin/tree/main). Each covers 50 tasks × 100 episodes with unseen instructions. Frozen snapshots are initial scenes and observations, not training demonstrations.

```bash
# From XPolicyLab/policy/Phoenix_MWA, in an online shell:
hf download AnyverseDynamics/Phoenix-RoboTwin \
  --revision 7c7326e077e84d443a9632080f466cda40fae88d \
  phoenix_robotwin_c2c_frozen_v1.tar.gz phoenix_robotwin_c2r_frozen_v1.tar.gz MANIFEST.json \
  --local-dir source/resources/robotwin/frozen
# Verify archive hashes against MANIFEST.json before extracting.
cd source/resources/robotwin/frozen
sha256sum phoenix_robotwin_*_frozen_v1.tar.gz
tar -xzf phoenix_robotwin_c2c_frozen_v1.tar.gz
tar -xzf phoenix_robotwin_c2r_frozen_v1.tar.gz
```

Expected archive SHA256: C2C `a58ec5a0fafb982031cee8d10ccda3578647c0593daf9c3a12a92291260d143f`; C2R `88c38c9361e411fd7c4a349ac8f518045538e2f981994fb2e4458f370265e2da`.

From the policy directory, after installing the pinned simulator and downloading the policy:

```bash
export ROBOTWIN_ROOT=/absolute/path/to/pinned/RoboTwin
export ROBOTWIN_SIM_PYTHON=/absolute/path/to/simulator/env/bin/python
export ROBOTWIN_FROZEN_C2C="$PWD/source/resources/robotwin/frozen/c2c"
export ROBOTWIN_FROZEN_C2R="$PWD/source/resources/robotwin/frozen/c2r"
export ROBOTWIN_OUTPUT_ROOT="$PWD/runs/frozen"
bash evaluate_frozen.sh "$PWD/checkpoints/phoenix-robotwin-35k" --mode c2c --gpus 0
bash evaluate_frozen.sh "$PWD/checkpoints/phoenix-robotwin-35k" --mode c2r --gpus 0
```

Add `--max-tasks 1 --episodes 1` for a bounded smoke. This optional route preserves the original evaluator. The submission entry point is `eval.sh` / the official RoboTwin `scripts/eval_policy.sh` documented in the main README.
