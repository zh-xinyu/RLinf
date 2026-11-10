# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-state evaluation helpers shared by RLinf workers and parity probes."""

import hashlib
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
HORIZONS = (240, 240, 320, 480)
EXECUTION = (5, 5, 5, 10)


def validate_selection(tasks: list[int], trials: list[int], envs: int) -> None:
    """Reject duplicate pairs and partial packets before creating simulators."""
    for values, limit in ((tasks, 40), (trials, 50)):
        if (
            not values
            or len(set(values)) != len(values)
            or any(
                type(value) is not int or value not in range(limit) for value in values
            )
        ):
            raise ValueError("Use unique nonempty official task/state indices")
    if type(envs) is not int or envs < 1 or len(trials) % envs:
        raise ValueError("envs-per-task must divide the selected state count")


def action_digest(actions: torch.Tensor) -> str:
    """Hash returned physical commands for exact paired rollout checks."""
    value = actions.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


class EpisodePolicy:
    """Run unchanged B1 inference with task/state-private Torch RNG streams.

    The seed matches the previous standalone evaluator (local task ID * 100 +
    state ID). The caller's Torch/NumPy/Python streams are restored after every
    prediction. Frozen slots receive their last finite action without inference.
    """

    def __init__(self, seed: int, trace: bool = False) -> None:
        self.seed = seed
        self.states = {}
        self.actions = {}
        self.trace = [] if trace else None
        self.calls = 0

    @torch.no_grad()
    def predict(
        self, model, obs: dict, keys: list[tuple[int, int]], active
    ) -> torch.Tensor:
        from rlinf.utils.nested_dict_process import split_dict_to_chunk

        keys = [tuple(map(int, key)) for key in keys]
        if len(keys) != len(set(keys)) or len(keys) != len(active):
            raise ValueError("Evaluation packet must contain unique episode keys")
        if model.training:
            raise ValueError("Evaluation requires model.eval()")
        rows = split_dict_to_chunk(obs, len(keys))
        results = []
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        for row, key, live in zip(rows, keys, active, strict=True):
            key = tuple(map(int, key))
            if bool(live):
                numpy_state, python_state = np.random.get_state(), random.getstate()
                try:
                    with torch.random.fork_rng(devices=devices):
                        if key in self.states:
                            cpu, cuda = self.states[key]
                            torch.set_rng_state(cpu)
                            if devices:
                                torch.cuda.set_rng_state(cuda, devices[0])
                        else:
                            torch.manual_seed(self.seed + (key[0] % 10) * 100 + key[1])
                        actions, _ = model.predict_action_batch(row, mode="eval")
                        actions = torch.as_tensor(actions).detach().cpu().contiguous()
                        if not torch.isfinite(actions).all():
                            raise FloatingPointError("Nonfinite evaluation action")
                        self.states[key] = (
                            torch.get_rng_state(),
                            torch.cuda.get_rng_state(devices[0]) if devices else None,
                        )
                        self.actions[key] = actions
                        self.calls += 1
                        if self.trace is not None:
                            self.trace.append(
                                {
                                    "task_id": key[0],
                                    "trial_id": key[1],
                                    "actions_sha256": action_digest(actions),
                                }
                            )
                finally:
                    np.random.set_state(numpy_state)
                    random.setstate(python_state)
            if key not in self.actions:
                raise ValueError("Frozen slot has no earlier live prediction")
            results.append(self.actions[key])
        return torch.cat(results)


def audit_episodes(rows: list[dict], tasks: list[int], trials: list[int]) -> dict:
    """Withhold scores unless every requested pair has one complete result."""
    validate_selection(tasks, trials, 1)
    expected = {(task, trial) for task in tasks for trial in trials}
    seen = set()
    for row in rows:
        key = row["task_id"], row["trial_id"]
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicate evaluation pair: {key}")
        seen.add(key)
        suite = key[0] // 10
        if (
            row["suite"] != SUITES[suite]
            or type(row["success_once"]) is not bool
            or not 1 <= row["simulation_steps"] <= HORIZONS[suite]
            or row["horizon"] != HORIZONS[suite]
            or (not row["success_once"] and row["simulation_steps"] != HORIZONS[suite])
        ):
            raise ValueError(f"Invalid or incomplete episode outcome: {key}")
    complete = seen == expected
    rates = (
        {
            suite: sum(row["success_once"] for row in rows if row["suite"] == suite)
            / sum(row["suite"] == suite for row in rows)
            for suite in SUITES
            if any(task // 10 == SUITES.index(suite) for task in tasks)
        }
        if complete
        else None
    )
    return {
        "complete": complete,
        "episodes": len(rows),
        "expected_episodes": len(expected),
        "missing_pairs": sorted(expected - seen),
        "suite_success_rates": rates,
        "macro_success_rate": sum(rates.values()) / len(rates) if complete else None,
    }


def write_json(path: Path, value: Any) -> None:
    """Atomically publish a local evaluation artifact."""
    import json

    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
