# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Deterministic physical-rollout allocation, independent of flow reuse."""

import math
from collections import Counter, deque
from typing import Any


def bounded_quotas(
    probabilities: list[float],
    total: int,
    previous: list[int],
    minimum: int,
    max_change: int,
) -> list[int]:
    """Project desired counts to feasible integers without losing trajectory slots."""
    if max_change < 0:
        raise ValueError("max_quota_change must be nonnegative.")
    lower = [max(minimum, n - max_change) for n in previous]
    upper = [min(total, n + max_change) for n in previous]
    if sum(lower) > total or sum(upper) < total:
        raise ValueError("Infeasible rollout quota bounds.")
    target = [p * total for p in probabilities]
    counts = lower.copy()
    while sum(counts) < total:
        eligible = [i for i in range(len(counts)) if counts[i] < upper[i]]
        chosen = max(eligible, key=lambda i: (target[i] - counts[i], -i))
        counts[chosen] += 1
    return counts


class RolloutAllocator:
    """Track task outcomes and produce plans with every suite on every worker.

    A plan has one entry per actual episode slot, with a task-local occurrence
    and reset seed. Shared prefixes of each task's reset sequence agree across
    allocation strategies even when tasks move to other workers.
    """

    def __init__(
        self,
        tasks: list[dict],
        workers: int,
        envs_per_worker: int,
        epochs: int,
        config: Any,
        reference: dict[str, float] | None = None,
    ) -> None:
        self.tasks = tasks
        self.workers = workers
        self.envs_per_worker = envs_per_worker
        self.epochs = epochs
        from omegaconf import OmegaConf

        self.config = (
            OmegaConf.to_container(config, resolve=True)
            if OmegaConf.is_config(config)
            else dict(config)
        )
        self.mode = self.config.get("mode", "adaptive")
        self.suites = list(dict.fromkeys(task["suite"] for task in tasks))
        self.pools = {
            suite: [task for task in tasks if task["suite"] == suite]
            for suite in self.suites
        }
        self.total = workers * envs_per_worker * epochs
        self.minimum = workers * epochs
        self.window_rounds = int(self.config.get("window_rounds", 20))
        self.history = deque(maxlen=2 * self.window_rounds)
        self.cursors = dict.fromkeys(self.suites, 0)
        self.occurrences = Counter()
        self.rounds = 0
        self.trajectories = 0
        self.simulation_steps = 0
        self.reference = reference
        self.last_diagnostics = {}
        if self.mode not in {"uniform", "static", "adaptive"}:
            raise ValueError("Unknown rollout allocation mode.")
        if not tasks or envs_per_worker < len(self.suites):
            raise ValueError("Each env worker needs at least one slot per suite.")
        if self.total % len(self.suites):
            raise ValueError("Initial uniform trajectory budget must divide by suites.")
        self.quotas = [self.total // len(self.suites)] * len(self.suites)
        for key in ("window_rounds", "interval_rounds", "min_task_episodes"):
            value = self.config.get(
                key,
                {"window_rounds": 20, "interval_rounds": 10, "min_task_episodes": 5}[
                    key
                ],
            )
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if int(self.config.get("warmup_rounds", 20)) < int(
            self.config.get("window_rounds", 20)
        ):
            raise ValueError("Allocation warmup must cover the success window.")
        rho = float(self.config.get("rho", 0.5))
        epsilon = float(self.config.get("epsilon", 0.01))
        if not 0 <= rho <= 1 or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Invalid allocation rho/epsilon.")
        if self.mode == "adaptive":
            if reference is None or set(reference) != set(self.suites):
                raise ValueError(
                    "Adaptive allocation needs calibrated teacher references."
                )
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in reference.values()):
                raise ValueError("Teacher success references must lie in [0, 1].")
        if self.mode == "static":
            counts = self.config.get("static_quotas", {})
            if set(counts) != set(self.suites) or sum(counts.values()) != self.total:
                raise ValueError("Static quotas must cover all suites and sum to T.")
            if any(not isinstance(v, int) or v < self.minimum for v in counts.values()):
                raise ValueError("Static quotas violate per-worker suite coverage.")

    def _decide(self) -> None:
        warmup = int(self.config.get("warmup_rounds", 20))
        interval = int(self.config.get("interval_rounds", 10))
        self.last_diagnostics = {"decision": False, "reason": "hold"}
        if self.mode == "uniform" or self.rounds < warmup:
            self.last_diagnostics["reason"] = "uniform_or_warmup"
            return
        if (self.rounds - warmup) % interval:
            return
        if self.mode == "static":
            self.quotas = [self.config["static_quotas"][s] for s in self.suites]
            self.last_diagnostics = {"decision": True, "reason": "static"}
            return
        success, counts, newest = Counter(), Counter(), {}
        current_window = list(self.history)[-self.window_rounds :]
        for records in current_window:
            for row in records:
                counts[row["task"]] += 1
                success[row["task"]] += int(row["success_once"])
                newest[row["task"]] = max(
                    newest.get(row["task"], 0), row.get("round", self.rounds)
                )
        minimum = int(self.config.get("min_task_episodes", 5))
        missing = [t["task"] for t in self.tasks if counts[t["task"]] < minimum]
        self.last_diagnostics["task_counts"] = dict(counts)
        self.last_diagnostics["task_latest_age_rounds"] = {
            t["task"]: self.rounds - newest[t["task"]] if t["task"] in newest else None
            for t in self.tasks
        }
        if missing:
            self.last_diagnostics.update(
                reason="insufficient_task_coverage", missing_tasks=missing
            )
            return
        estimates = {
            s: sum(success[t["task"]] / counts[t["task"]] for t in self.pools[s])
            / len(self.pools[s])
            for s in self.suites
        }
        # Signed learning progress is diagnostic only; it never changes allocation.
        if len(self.history) == 2 * self.window_rounds:
            old_success, old_counts = Counter(), Counter()
            for records in list(self.history)[: self.window_rounds]:
                for row in records:
                    old_counts[row["task"]] += 1
                    old_success[row["task"]] += int(row["success_once"])
            if all(old_counts[t["task"]] >= minimum for t in self.tasks):
                self.last_diagnostics["signed_progress"] = {
                    s: estimates[s]
                    - sum(
                        old_success[t["task"]] / old_counts[t["task"]]
                        for t in self.pools[s]
                    )
                    / len(self.pools[s])
                    for s in self.suites
                }
        gap = {s: max(0.0, self.reference[s] - estimates[s]) for s in self.suites}
        rho, eps = (
            float(self.config.get("rho", 0.5)),
            float(self.config.get("epsilon", 0.01)),
        )
        normalizer = sum(gap.values()) + len(self.suites) * eps
        probabilities = [
            (1 - rho) / len(self.suites) + rho * (gap[s] + eps) / normalizer
            for s in self.suites
        ]
        previous_quotas = self.quotas.copy()
        self.quotas = bounded_quotas(
            probabilities,
            self.total,
            self.quotas,
            self.minimum,
            int(self.config.get("max_quota_change", 8)),
        )
        self.last_diagnostics.update(
            decision=True,
            reason="gap",
            success=estimates,
            gap=gap,
            probabilities=dict(zip(self.suites, probabilities)),
            desired_quotas=dict(
                zip(self.suites, [p * self.total for p in probabilities])
            ),
            quota_delta=dict(
                zip(
                    self.suites,
                    [n - old for n, old in zip(self.quotas, previous_quotas)],
                )
            ),
            minimum_quota=self.minimum,
        )

    def plan(self) -> list[dict]:
        """Allocate the next round; must be called exactly once per observed round."""
        if getattr(self, "pending", None) is not None:
            raise RuntimeError(
                "Cannot generate a second plan before observing its result."
            )
        self._decide()
        k = len(self.suites)
        cells = self.workers * self.epochs
        local = [[1] * k for _ in range(cells)]
        remaining = [n - cells for n in self.quotas]
        # Rotate placement to avoid permanently assigning expensive tasks to a rank.
        for offset in range(cells):
            cell = (offset + self.rounds) % cells
            for _ in range(self.envs_per_worker - k):
                suite = max(range(k), key=lambda i: (remaining[i], -((i - cell) % k)))
                if remaining[suite] <= 0:
                    raise RuntimeError("Quota placement exhausted prematurely.")
                local[cell][suite] += 1
                remaining[suite] -= 1
        if any(remaining):
            raise RuntimeError("Quota placement failed to preserve total counts.")
        rows = []
        base_seed = int(self.config.get("seed", 0))
        for cell, counts in enumerate(local):
            epoch, worker = divmod(cell, self.workers)
            slot = 0
            for i, suite in enumerate(self.suites):
                for _ in range(counts[i]):
                    task = self.pools[suite][
                        self.cursors[suite] % len(self.pools[suite])
                    ]
                    self.cursors[suite] += 1
                    occurrence = self.occurrences[task["task"]]
                    self.occurrences[task["task"]] += 1
                    seed = (base_seed + task["task_id"] * 1000003 + occurrence) % (
                        2**32
                    )
                    rows.append(
                        dict(
                            task,
                            round=self.rounds + 1,
                            epoch=epoch,
                            worker=worker,
                            slot=slot,
                            occurrence=occurrence,
                            reset_seed=seed,
                        )
                    )
                    slot += 1
        self.pending = rows
        return rows

    def observe(self, records: list[dict]) -> None:
        """Validate physical episode records before updating the controller."""
        if getattr(self, "pending", None) is None or len(records) != self.total:
            raise ValueError("Missing plan or incorrect physical trajectory count.")

        def key(r):
            return (r["round"], r["epoch"], r["worker"], r["slot"])

        actual = {key(row): row for row in records}
        if len(actual) != len(records):
            raise ValueError("Duplicate episode records.")
        for planned in self.pending:
            row = actual.get(key(planned))
            if row is None or any(
                row.get(f) != planned[f]
                for f in ("task", "suite", "task_id", "occurrence", "reset_seed")
            ):
                raise ValueError("Episode identity does not match the rollout plan.")
            steps = row.get("simulation_steps")
            if not isinstance(steps, int) or not 0 < steps <= planned["horizon"]:
                raise ValueError("Invalid real environment step count.")
            if not isinstance(row.get("success_once"), bool):
                raise ValueError("Missing episode success outcome.")
            if not row["success_once"] and steps != planned["horizon"]:
                raise ValueError("Failed episode ended before the native task horizon.")
        self.history.append(records)
        self.rounds += 1
        self.trajectories += len(records)
        self.simulation_steps += sum(row["simulation_steps"] for row in records)
        self.pending = None

    def state_dict(self) -> dict:
        """Export controller history and cursors for provenance, not exact resume."""
        return {
            "rounds": self.rounds,
            "trajectories": self.trajectories,
            "simulation_steps": self.simulation_steps,
            "quotas": dict(zip(self.suites, self.quotas)),
            "cursors": self.cursors,
            "occurrences": dict(self.occurrences),
            "history": list(self.history),
            "diagnostics": self.last_diagnostics,
            "config": self.config,
            "reference": self.reference,
        }


def training_signature(cfg: Any) -> dict:
    """Freeze observation, action and rollout semantics used by calibration."""
    from omegaconf import OmegaConf

    env = OmegaConf.to_container(cfg.env.train, resolve=True)
    # HF rollout copies the actor model, then overrides rollout loading settings.
    model = OmegaConf.to_container(cfg.actor.model, resolve=True)
    model.update(OmegaConf.to_container(cfg.rollout.model, resolve=True))
    env_keys = (
        "env_type",
        "task_source",
        "dataset_source",
        "split",
        "task_soup",
        "task_filter",
        "task_mode",
        "episode_horizon_source",
        "auto_reset",
        "ignore_terminations",
        "init_params",
        "observation",
        "action_space",
    )
    model_keys = (
        "model_type",
        "precision",
        "action_head_precision",
        "num_action_chunks",
        "action_dim",
        "denoising_steps",
        "obs_converter_type",
        "embodiment_tag",
        "policy_setup",
        "rl_head_config",
    )
    return {
        "env": {k: env.get(k) for k in env_keys},
        "model": {k: model.get(k) for k in model_keys},
        "teacher_action_head_precision": {
            name: expert.get(
                "action_head_precision", model.get("action_head_precision")
            )
            for name, expert in cfg.rollout.expert_models.items()
        },
    }


def load_teacher_reference(path: str, cfg: Any, tasks: list[dict]) -> dict[str, float]:
    """Reject incomplete calibration, different tasks, or mismatched semantics."""
    import json
    from pathlib import Path

    reference = json.loads(Path(path).read_text())
    if reference.get("schema_version") != 1 or reference.get("status") != "COMPLETE":
        raise ValueError("Teacher calibration must be complete (schema_version=1).")
    if reference["training_signature"] != training_signature(cfg):
        raise ValueError("Teacher calibration does not match the training protocol.")
    expected = {(t["task"], t["suite"], t["horizon"]) for t in tasks}
    actual = {(t["task"], t["suite"], t["horizon"]) for t in reference["tasks"]}
    if actual != expected or len(reference["tasks"]) != len(tasks):
        raise ValueError("Teacher calibration task set/horizons do not match.")
    teacher_paths = {
        str(name): str(Path(e.model_path).resolve())
        for name, e in cfg.rollout.expert_models.items()
    }
    if reference["teacher_checkpoints"] != teacher_paths:
        raise ValueError("Teacher calibration uses different checkpoints.")
    values = {}
    for task in reference["tasks"]:
        n, successes = task["episodes"], task["successes"]
        if (
            type(n) is not int
            or type(successes) is not int
            or n < 1
            or not 0 <= successes <= n
        ):
            raise ValueError("Invalid teacher episode counts.")
        values.setdefault(task["suite"], []).append(successes / n)
    return {suite: sum(rates) / len(rates) for suite, rates in values.items()}
