# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Task catalog and reset sampling for uniform multi-suite LIBERO training."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch


class CombinedLiberoBenchmark:
    """Expose several benchmarks as one stable, suite-labelled task catalog.

    Task identity is the pair (suite name, local task ID). Task names are not
    deduplicated: the same named task in different suites retains its own label.
    """

    def __init__(self, suite_names: Sequence[str], factory: Callable[[str], Any]):
        self.suite_names = tuple(str(name) for name in suite_names)
        if not self.suite_names or len(set(self.suite_names)) != len(self.suite_names):
            raise ValueError("task_suite_names must be nonempty and unique")
        self.benchmarks = {name: factory(name) for name in self.suite_names}
        self.task_refs = [
            (name, task_id)
            for name, benchmark in self.benchmarks.items()
            for task_id in range(benchmark.get_num_tasks())
        ]
        if any(
            benchmark.get_num_tasks() <= 0 for benchmark in self.benchmarks.values()
        ):
            raise ValueError("Each LIBERO suite must contain at least one task")

    def _reference(self, task_id: int) -> tuple[str, int]:
        if not 0 <= int(task_id) < len(self.task_refs):
            raise IndexError(f"Global LIBERO task ID out of range: {task_id}")
        return self.task_refs[int(task_id)]

    def get_num_tasks(self) -> int:
        """Return the total number of suite-labelled tasks."""
        return len(self.task_refs)

    def get_task(self, task_id: int) -> Any:
        """Return the source task, preserving its BDDL folder and language."""
        suite, local_id = self._reference(task_id)
        return self.benchmarks[suite].get_task(local_id)

    def get_task_init_states(self, task_id: int) -> Any:
        """Load official NumPy initial states with a narrow PyTorch allowlist."""
        suite, local_id = self._reference(task_id)
        with torch.serialization.safe_globals(
            [
                np.core.multiarray._reconstruct,
                np.ndarray,
                np.dtype,
                type(np.dtype(np.float64)),
                type(np.dtype(np.float32)),
            ]
        ):
            return self.benchmarks[suite].get_task_init_states(local_id)

    def suite_for_task(self, task_id: int) -> str:
        """Return the hard-router label for a global task ID."""
        return self._reference(task_id)[0]


class SuiteBalancedResetSampler:
    """Draw equal suite quotas and cycle all tasks within each suite.

    Tasks are reshuffled after each complete traversal; initial-state indices are
    sampled uniformly for the chosen task. Every full draw includes every suite,
    even if suites have different task counts or different numbers of trials.
    """

    def __init__(
        self,
        suite_labels: Sequence[str],
        trial_counts: Sequence[int],
        generator: np.random.Generator,
        task_filter: Sequence[int] | None = None,
    ):
        if len(suite_labels) != len(trial_counts) or not suite_labels:
            raise ValueError(
                "Task labels and trial counts must have equal nonzero length"
            )
        self.suites = tuple(dict.fromkeys(suite_labels))
        self.trial_counts = np.asarray(trial_counts, dtype=np.int64)
        if np.any(self.trial_counts <= 0):
            raise ValueError("Every task must have at least one initial state")
        self.offsets = np.concatenate(([0], np.cumsum(self.trial_counts)[:-1]))
        self.generator = generator
        selected = set(range(len(suite_labels)) if task_filter is None else task_filter)
        if not selected.issubset(range(len(suite_labels))):
            raise ValueError("task_filter contains an invalid global task ID")
        self.tasks = {
            suite: np.asarray(
                [
                    i
                    for i, label in enumerate(suite_labels)
                    if label == suite and i in selected
                ],
                dtype=np.int64,
            )
            for suite in self.suites
        }
        if any(len(tasks) == 0 for tasks in self.tasks.values()):
            raise ValueError("Suite-balanced task filters must retain every suite")
        self.orders = {
            suite: generator.permutation(tasks) for suite, tasks in self.tasks.items()
        }
        self.cursors = dict.fromkeys(self.suites, 0)

    def sample(self, count: int) -> np.ndarray:
        """Return reset IDs, interleaving equal per-suite quotas."""
        if count <= 0 or count % len(self.suites):
            raise ValueError(
                "Suite-balanced reset count must be positive and divisible by suites"
            )
        result = []
        for _ in range(count // len(self.suites)):
            for suite in self.suites:
                if self.cursors[suite] == len(self.orders[suite]):
                    self.orders[suite] = self.generator.permutation(self.tasks[suite])
                    self.cursors[suite] = 0
                task = int(self.orders[suite][self.cursors[suite]])
                self.cursors[suite] += 1
                trial = int(self.generator.integers(self.trial_counts[task]))
                result.append(int(self.offsets[task]) + trial)
        return np.asarray(result, dtype=np.int64)
