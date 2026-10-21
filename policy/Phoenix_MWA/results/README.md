# RoboTwin 2.0 results

Phoenix-MWA: 230,000-step initialization and 35,000-step posttraining. Each condition covers 50 tasks with 100 episodes per task.

| Condition | Successes | Episodes | Success rate |
| --- | ---: | ---: | ---: |
| Clean to clean (C2C) | 4,327 | 5,000 | 86.54% |
| Clean to random (C2R) | 3,621 | 5,000 | 72.42% |
| Average | | | 79.48% |

- [Per-task CSV](robotwin_task_results.csv): 50 rows with successes, episode counts, success rates (%) and the mean of C2C/C2R for each task.
- [Original result archive](Phoenix_C2C_C2R_100_task_logs.zip): 100 client logs and their index, preserved unchanged. CSV counts were exported from the supplied `task_logs.json` and checked against its totals.
- [Evaluation protocol](../ROBOTWIN_ORIGINAL.md): these scores come from the original frozen-scene evaluation, rather than a rerun through the current XPolicyLab adapter.

The current XPolicyLab adapter passed both GPU debug modes on 2026-10-21: 10 plain-RGB episodes and 10 encoded-observation episodes, each ending with `[MAIN] eval finished` and exit code 0. The released 35K checkpoint and its dependencies matched the download manifest.
