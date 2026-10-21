# RoboTwin source evaluation evidence — 2026-10-18 bundle

User-supplied results for the Phoenix model reported as 230000-step initialization + 35000-step posttraining, imported on 2026-10-21. These are historical source-protocol runs, not a new evaluation of the XPolicyLab adapter in PR #187. Run paths identify C2C on 2026-10-17 and C2R on 2026-10-16.

| Protocol | Tasks | Completed episodes | Successes | Success rate |
| --- | ---: | ---: | ---: | ---: |
| Clean to clean (C2C) | 50 | 5000 | 4327 | 86.54% |
| Clean to random (C2R) | 50 | 5000 | 3621 | 72.42% |

- [Task-by-task log index](逐任务日志索引.md): links to all 100 client logs.
- [Original structured results](task_logs.json): per-task counts, embedded client logs, SHA256 hashes and source run paths.
- [Summary](summary.json): recomputed totals and the tasks whose logs contain tracebacks.
- [Original log archive](Phoenix_C2C_C2R_100_task_logs.zip) and [HTML index](index.html).
- [Checksums](SHA256SUMS.txt): SHA256 of the supplied logs, archive and structured results.

All 100 client-log files match their SHA256 and embedded text in `task_logs.json`; each task is marked complete with 100 episodes, and both totals match the sum of the task records. All logs contain a result-save marker. The client logs themselves generally do not print episode success counts: the counts above come from the supplied JSON. The referenced `_result_clean.txt` / `_result_random.txt` files are not included, so per-episode outcomes were not independently reconstructed.

The original logs are retained unchanged, including recovery traces: 3 C2C logs and 10 C2R logs contain tracebacks followed by result-save markers. This bundle is not represented as an error-free run. See `summary.json` for the affected tasks. The Markdown index's machine-local links were changed to relative links, and the HTML index's line endings were normalized. Original client logs, structured results and ZIP bytes are unchanged.

This evidence does not verify the XPolicyLab debug client with both plain and encoded observations, establish a current-adapter success rate, or pin the loaded checkpoint by hash (the archive has no model-load receipt). Those claims require separate validation.
