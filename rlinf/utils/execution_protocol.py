# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Decision scheduling for independent LIBERO environments with different prefixes."""

import math


def suite_execution_lengths(env_cfg, storage_width: int) -> dict[str, int] | None:
    """Validate opt-in variable execution; leave legacy fixed-width runs unchanged."""
    configured = env_cfg.get("task_suite_execution_lengths", None)
    if configured is None:
        return None
    names = list(env_cfg.get("task_suite_names", []))
    horizons = env_cfg.get("task_suite_horizons", {})
    if (
        env_cfg.get("env_type") != "libero"
        or not names
        or set(configured) != set(names)
        or set(horizons) != set(names)
    ):
        raise ValueError("Mixed execution requires every selected LIBERO suite/horizon")
    if (
        not env_cfg.get("freeze_finished_episodes", False)
        or env_cfg.get("auto_reset", False)
        or env_cfg.get("ignore_terminations", False)
    ):
        raise ValueError("Mixed execution requires frozen, non-autoreset full episodes")
    values = dict(configured)
    if any(type(k) is not int or k < 1 or k > storage_width for k in values.values()):
        raise ValueError("Suite execution lengths must fit the common action width")
    if max(values.values()) != storage_width:
        raise ValueError("Common action width must equal the largest executed prefix")
    if any(type(horizons[s]) is not int or horizons[s] < values[s] for s in names):
        raise ValueError("Each suite needs a positive compatible physical horizon")
    return values


def rollout_decision_steps(env_cfg, model_cfg) -> int:
    """Return a shared decision count, allowing different per-slot physical clocks."""
    width = int(model_cfg.num_action_chunks)
    lengths = suite_execution_lengths(env_cfg, width)
    if lengths is None:
        return int(env_cfg.max_steps_per_rollout_epoch) // width
    return max(
        math.ceil(int(env_cfg.task_suite_horizons[s]) / k) for s, k in lengths.items()
    )


def mask_executed_prefix(mask, teacher_ids, teacher_names, configured):
    """Exclude storage padding from terminal masks and physical-control accounting."""
    if configured is None:
        return mask
    import torch

    ids = teacher_ids[: mask.shape[0]].reshape(mask.shape[:2]).to(mask.device).long()
    lengths = torch.tensor(
        [configured[name] for name in teacher_names], device=mask.device
    )[ids]
    positions = torch.arange(mask.shape[-1], device=mask.device)
    return mask & (positions < lengths[..., None])
