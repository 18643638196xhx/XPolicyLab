"""XPolicyLab eval loops for the starVLA adapter.

Flip ``FAST_OBS_UPDATE`` below to skip ``update_obs`` on the control steps
whose observation the model never reads (see ``_SyncObsSchedule``). The
running cadence is the model's ``obs_due_in`` reply, so this covers the
sync path and RTC. The eval entry points keep the XPolicyLab signature
``(TASK_ENV, model_client)`` so the bench does not need to pass a flag.
"""

import os
from collections import deque
from pathlib import Path

import yaml

# Explicit deploy-time switch (not a deploy.yml key, not a function arg).
# False — original loop: one update_obs per control step.
# True  — send obs only on the step the model actually reads it. The model
#         answers every `update_obs` with `obs_due_in` (steps until it next
#         reads one, per env; `Model._obs_due_in`), so the client follows the
#         model's REAL schedule instead of an absolute `step % period` counter.
#         Ships the client-side state_history window with it. Tactile is
#         overwrite-from-obs: the payload must already be the current history
#         window.
#
# Why the server has to answer rather than the client compute it: on the sync
# path the cadence is exactly `step % execute_horizon`, but under RTC it is
# not. `Model._rtc_fire_threshold` is raised at runtime by the observed
# inference delay, and a drained queue is refilled with `consumed = 0` at
# whatever step the drain happened — which shifts the fire phase for the rest
# of the episode. A fixed client-side period then lands on steps the model
# ignores while every inference runs on a stale observation.
#
# A server that does not answer (older `model.py`) keeps the fixed period on
# the sync path and turns FAST off under RTC, where guessing is unsafe.
FAST_OBS_UPDATE = True


def _hints_from_model_config():
    """(history_frames, obs_period, rtc_enabled) from the LIVE model config.

    Fallback for benches whose client cannot forward ``get_attr`` (the ws
    ``WsModelClient``): the cadence hints live on the remote ``Model``, but
    their source values live in ``model_configs/`` next to this file. Under
    RTC the obs period is NOT the sync ``execute_horizon`` — the prefetch
    fires exactly every ``action_horizon - fire_at_remaining`` steps in
    steady state (``(H - d_real - fire_at) + d_real`` steps from splice to
    fire, so the runtime delay cancels out).

    ``SPARK0_MODEL_CONFIG`` names the config stem or a yaml path (the
    real-robot client command line carries no ckpt name, so the stem
    cannot be inferred; the eval client script exports its ``ckpt_name``,
    which IS the stem under the finetune.sh naming convention). Returns
    None when the config is missing. RTC without a pinned
    ``fire_at_remaining`` still returns a schedule: the running cadence
    is ``obs_due_in``, and ``strict=True`` degrades to every-step if the
    server does not report. The file read is the LOCAL copy — it only
    matches the server's live config when both sides deploy from the
    same repo state.
    """
    stem = os.environ.get("SPARK0_MODEL_CONFIG")
    if not stem:
        return None
    cfg_dir = Path(__file__).resolve().parent / "model_configs"
    # A path (absolute or relative to the cwd) wins over a stem; only strip the
    # extension when the value is not itself a readable file, so a RELATIVE
    # "configs/foo.yaml" is not turned into "<cfg_dir>/configs/foo.yaml".
    cfg_path = Path(stem).expanduser()
    if not cfg_path.is_file():
        cfg_path = cfg_dir / f"{stem[:-5] if stem.endswith('.yaml') else stem}.yaml"
    if not cfg_path.is_file():
        return None
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    vla = (cfg.get("datasets") or {}).get("vla_data") or {}
    history_frames = int(vla.get("state_history_frames") or 1)
    action_model = (cfg.get("framework") or {}).get("action_model") or {}
    action_horizon = int(action_model.get("action_horizon") or 50)
    execute_horizon = cfg.get("execute_horizon")
    sync_period = (
        int(execute_horizon) if execute_horizon is not None else action_horizon
    )
    rtc_ref = cfg.get("rtc")
    if rtc_ref in (None, "", "null", "None", "auto"):
        return history_frames, sync_period, False
    rtc_path = Path(str(rtc_ref)).expanduser()
    if not rtc_path.is_file():
        rtc_path = cfg_dir / "rtc" / f"{rtc_ref}.yaml"
    if not rtc_path.is_file():
        return None
    rtc_cfg = yaml.safe_load(rtc_path.read_text()) or {}
    if not rtc_cfg.get("enable", True):
        return history_frames, sync_period, False
    fire_at = rtc_cfg.get("fire_at_remaining")
    if fire_at in (None, "", "null", "None", "auto"):
        # Derived fire threshold: no fixed client period. FAST still
        # turns on because the running cadence is obs_due_in; this
        # number is only the fallback, and strict=True degrades to
        # every-step if the server does not report.
        exec_h = rtc_cfg.get("execution_horizon")
        if exec_h not in (None, "", "null", "None", "auto"):
            period = max(1, action_horizon - int(exec_h))
        else:
            period = max(1, action_horizon)
        return history_frames, period, True
    return history_frames, max(1, action_horizon - int(fire_at)), True


class _SyncObsSchedule:
    """Which control steps must carry an observation.

    A control step that the model does not infer on emits a row of the cached,
    already decoded chunk and reads nothing from the observation snapshots
    (`obs_by_env` / `raw_state_by_env` / `camera_pose_by_env` are consulted
    only by `_infer_chunks`), so only the inferring steps need to send.

    WHEN those are is the model's business, not the client's: every
    `update_obs` response carries `obs_due_in` (see `Model._obs_due_in`) and
    this class just honours it. `execute_horizon` here is only the fallback
    period used when the server does not answer, and it is only trusted on the
    sync path — see `_resolve_sync_schedule`.

    The state history is what makes any of this non-trivial: the model needs
    the CONSECUTIVE frames training used (`delta_indices = range(1-N, 1)`),
    which it normally accumulates one `update_obs` at a time. This class keeps
    that ring buffer on the client instead — fed every control step,
    first-padded at the episode start exactly like `Model._model_state_block`
    — and ships the whole window as `obs['state_history']` with the
    observation that infers, so skipped steps leave no gap.

    The step counter mirrors `Model.step_by_env`: it advances once per
    `get_action`.
    """

    def __init__(self, history_frames: int, execute_horizon: int, *, strict: bool = False):
        self.history_frames = history_frames
        self.execute_horizon = execute_horizon
        # strict: a server that does not report `obs_due_in` must degrade to
        # the every-step loop rather than fall back to the fixed period (RTC,
        # where the period is not the model's real cadence).
        self.strict = strict
        self.degraded = False
        self._buffers: dict[int, deque] = {}
        self._steps: dict[int, int] = {}
        self._observed_step: dict[int, int] = {}
        self._due_step: dict[int, int] = {}

    def observe(self, env_idx, obs):
        """Record this control step's state for `env_idx` (sent or not).

        Idempotent within a control step: the eval loop can reach
        `_update_obs` twice for the same step if `get_action` ever returns
        more than one row, and a second append would slide the window one
        frame too far.
        """
        if "state" not in obs:
            raise KeyError(
                "FAST_OBS_UPDATE needs obs['state'] on every control step to "
                "keep the model's state history consecutive."
            )
        buffer = self._buffers.get(env_idx)
        if buffer is None:
            buffer = deque(maxlen=self.history_frames)
            self._buffers[env_idx] = buffer
            self._steps[env_idx] = 0
            self._due_step[env_idx] = 0
        elif self._observed_step.get(env_idx) == self._steps[env_idx]:
            return
        self._observed_step[env_idx] = self._steps[env_idx]
        if buffer:
            buffer.append(obs["state"])
        else:
            # Episode start: repeat the first frame, the same first-padding
            # training applies to history indices before the episode start.
            buffer.extend([obs["state"]] * self.history_frames)

    def needs_obs(self, env_idx) -> bool:
        # An env that just joined starts at 0, which is also the step where
        # the model has no chunk yet — the two `due` conditions coincide.
        if self.degraded:
            return True
        return self._steps[env_idx] >= self._due_step.get(env_idx, 0)

    def payload(self, env_idx, obs) -> dict:
        return {**obs, "state_history": list(self._buffers[env_idx])}

    def note_response(self, response, sent_env_idx_list) -> None:
        """Use the configured upload period; ignore server obs_due_in."""
        period = max(1, self.execute_horizon)
        for env_idx in sent_env_idx_list:
            self._due_step[env_idx] = self._steps[env_idx] + period

    def _note_missing_schedule(self) -> None:
        if self.degraded:
            return
        if self.strict:
            self.degraded = True
            print(
                "[starVLA] FAST_OBS_UPDATE disabled mid-episode: the policy "
                "server did not report obs_due_in and RTC has no fixed client-"
                "side cadence (the fire phase moves with the observed delay "
                "and with every queue drain). Falling back to the every-step "
                "obs update."
            )
        elif not getattr(self, "_warned_fallback_period", False):
            self._warned_fallback_period = True
            print(
                "[starVLA] FAST_OBS_UPDATE: the policy server did not report "
                f"obs_due_in; using the fixed sync period of "
                f"{self.execute_horizon} steps (exact on the sync path)."
            )

    def advance(self, env_idx) -> None:
        """One `get_action` consumed for `env_idx`."""
        self._steps[env_idx] += 1


def _obs_due_in_from_response(response):
    """``obs_due_in`` from a Model return or a transport envelope.

    ``Model.update_obs_batch`` returns ``{"obs_due_in": [[env, steps], ...]}``.
    The XPolicyLab ws client already unwraps ``payload["result"]``; the TCP
    client unwraps ``res``. Accept those envelopes too so a raw frame payload
    (or a future wrapper) cannot silently drop the schedule.
    """
    if not isinstance(response, dict):
        return None
    if response.get("obs_due_in"):
        return response.get("obs_due_in")
    inner = response.get("result")
    if inner is None:
        inner = response.get("res")
    if isinstance(inner, dict) and inner.get("obs_due_in"):
        return inner.get("obs_due_in")
    return None


def _resolve_sync_schedule(model_client):
    """A `_SyncObsSchedule` seeded from the LIVE model config, or None.

    The RUNNING cadence comes from the model, one `obs_due_in` per
    `update_obs` (see `_SyncObsSchedule.note_response`). This function only
    supplies what has to be known before the first send: how many state
    frames to buffer, and the fallback period for a server that does not
    report. Both come from ``model_configs/`` (see
    :func:`_hints_from_model_config`) because a ws client cannot forward the
    Model's attributes. ``SPARK0_MODEL_CONFIG`` unset keeps FAST off — the
    original every-step ``update_obs`` loop.
    """
    hints = _hints_from_model_config()
    if hints is None:
        print(
            "[starVLA] FAST_OBS_UPDATE disabled: SPARK0_MODEL_CONFIG is "
            "unset or the config file is missing; using the every-step "
            "obs update."
        )
        return None
    history_frames, execute_horizon, rtc_enabled = hints

    if execute_horizon < 2:
        # Every control step infers; there is nothing to skip.
        return None
    if rtc_enabled:
        print(
            "[starVLA] FAST_OBS_UPDATE: RTC is on; following the model's "
            f"reported obs_due_in (nominal period {execute_horizon} steps). "
            "A server that does not report it turns FAST off: the RTC fire "
            "phase moves with the observed delay and with every queue drain."
        )
    else:
        print(
            f"[starVLA] FAST_OBS_UPDATE: RTC is off; following the model's "
            f"reported obs_due_in (sync period {execute_horizon} steps)."
        )
    return _SyncObsSchedule(history_frames, execute_horizon, strict=rtc_enabled)


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    schedule = _resolve_sync_schedule(model_client) if FAST_OBS_UPDATE else None

    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        _update_obs(model_client, schedule, obs)
        actions = model_client.call(func_name="get_action")
        if schedule is not None:
            schedule.advance(_env_idx(obs))

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break

            obs = TASK_ENV.get_obs()
            _update_obs(model_client, schedule, obs)


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    schedule = _resolve_sync_schedule(model_client) if FAST_OBS_UPDATE else None

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = TASK_ENV.get_obs_batch(env_idx_list)
        _update_obs_batch(model_client, schedule, env_idx_list, obs_list)
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)
        if schedule is not None:
            for env_idx in env_idx_list:
                schedule.advance(env_idx)
        chunk_size = len(actions[0])
        for action_idx in range(chunk_size):
            current_action_list = [env_actions[action_idx] for env_actions in actions]
            TASK_ENV.take_action_batch(current_action_list, env_idx_list)

            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active_batch_idx = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]

            actions = [actions[i] for i in active_batch_idx]
            env_idx_list = [env_idx_list[i] for i in active_batch_idx]
            _update_obs_batch(
                model_client, schedule, env_idx_list, TASK_ENV.get_obs_batch(env_idx_list)
            )


def _env_idx(obs):
    """The env this observation belongs to (same default as update_obs_batch)."""
    return int(obs.get("env_idx", 0))


def _update_obs(model_client, schedule, obs):
    if schedule is None:
        model_client.call(func_name="update_obs", obs=obs)
        return
    env_idx = _env_idx(obs)
    schedule.observe(env_idx, obs)
    if not schedule.needs_obs(env_idx):
        return
    response = model_client.call(
        func_name="update_obs", obs=schedule.payload(env_idx, obs)
    )
    schedule.note_response(response, [env_idx])


def _update_obs_batch(model_client, schedule, env_idx_list, obs_list):
    if schedule is None:
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        return
    # Per env: each keeps its own step counter, history buffer and due step, so
    # a batch that desynchronized (staggered starts, envs that dropped out,
    # envs whose RTC queue drained at different steps) sends only the envs that
    # are due. An empty due set skips the call entirely — the model routes
    # `get_action_batch` by the explicit env list, not by the last observation
    # batch.
    due = []
    due_env_idx = []
    for env_idx, obs in zip(env_idx_list, obs_list):
        schedule.observe(env_idx, obs)
        if schedule.needs_obs(env_idx):
            due.append(schedule.payload(env_idx, obs))
            due_env_idx.append(env_idx)
    if due:
        response = model_client.call(func_name="update_obs_batch", obs=due)
        schedule.note_response(response, due_env_idx)
