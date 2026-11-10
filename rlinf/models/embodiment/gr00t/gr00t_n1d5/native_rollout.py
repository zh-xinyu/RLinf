# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Native-input rollout buckets with the original per-environment noise stream."""

from typing import Any

import torch


def native_rollout_plan(inputs: list[dict], maximum_batch_size: int) -> list[list[int]]:
    """Group matching prepared B1 shapes without adding text or duplicate rows."""
    if type(maximum_batch_size) is not int or maximum_batch_size < 1 or not inputs:
        raise ValueError(
            "Native rollout needs nonempty inputs and a positive integer batch size"
        )
    buckets = {}
    for index, row in enumerate(inputs):
        if not row or any(not isinstance(v, torch.Tensor) for v in row.values()):
            raise TypeError("Prepared native rollout inputs must contain only tensors")
        ids, mask = row["eagle_input_ids"], row["eagle_attention_mask"]
        if (
            ids.ndim != 2
            or ids.shape[0] != 1
            or mask.shape != ids.shape
            or not mask.bool().all()
        ):
            raise ValueError(
                "Prepare each native observation independently without text padding"
            )
        signature = tuple(
            (key, tuple(value.shape), value.dtype, value.device)
            for key, value in sorted(row.items())
        )
        buckets.setdefault(signature, []).append(index)
    return [
        indices[start : start + maximum_batch_size]
        for indices in buckets.values()
        for start in range(0, len(indices), maximum_batch_size)
    ]


def pack_native_rollout_inputs(inputs: list[dict], indices: list[int]) -> dict:
    """Concatenate B1 rows, including flattened per-image processor tensors."""
    return {
        key: torch.cat([inputs[i][key] for i in indices], dim=0)
        for key in inputs[indices[0]]
    }


def sample_native_rollout_noise(
    head: Any, rows: int, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Replay B1 initial/step draws in environment order before bucket execution.

    ODE currently draws a noise tensor even for each zero-variance transition.
    Preserve these draws as well as their dtype and shape; changing only the
    grouping must not change the worker RNG state or its next rollout's noise.
    """
    if head.rl_config.noise_method != "flow_ode":
        raise ValueError("Native rollout batching currently supports flow_ode only")
    shape = (1, head.config.action_horizon, head.config.action_dim)
    initial, transitions = [], []
    for _ in range(rows):
        initial.append(torch.randn(shape, device=device, dtype=dtype))
        transitions.append(
            torch.stack(
                [
                    head.sample_noise(shape, device)
                    for _ in range(head.num_inference_timesteps)
                ],
                dim=1,
            )
        )
    return {
        "initial": torch.cat(initial, dim=0),
        "transitions": torch.cat(transitions, dim=0),
    }


def slice_rollout_result(value: Any, row: int) -> Any:
    """Recover one environment's complete nested action/flow/storage payload."""
    if isinstance(value, dict):
        return {key: slice_rollout_result(item, row) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value[row : row + 1]
    if value is None:
        return None
    raise TypeError(f"Unsupported native rollout result type: {type(value)}")


def native_head_rows(
    head: Any, backbone: Any, actions: Any, mode: str, rollout_noise: Any
) -> tuple:
    """Keep B1 flow-head kernels while batching the more expensive backbone."""
    from transformers.feature_extraction_utils import BatchFeature

    predicted, outputs = [], []
    for row in range(backbone.backbone_features.shape[0]):

        def row_noise(device, dtype, row=row):
            batch = (
                rollout_noise(device, dtype)
                if callable(rollout_noise)
                else rollout_noise
            )
            return slice_rollout_result(batch, row)

        prediction, output = head.get_rl_action(
            BatchFeature(data=slice_rollout_result(dict(backbone), row)),
            BatchFeature(data=slice_rollout_result(dict(actions), row)),
            mode=mode,
            **({"rollout_noise": row_noise} if rollout_noise is not None else {}),
        )
        predicted.append(dict(prediction))
        outputs.append(output)

    def concatenate(values):
        return {key: torch.cat([value[key] for value in values]) for key in values[0]}

    return BatchFeature(data=concatenate(predicted)), concatenate(outputs)
