# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Strict held-out replay through the same environment used by MOPD workers."""

import copy
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.envs.robotwin.multitask_env import RoboTwinMultiTaskEnv
from toolkits.robotwin_mopd.author_pool import (
    load_pool,
    validate_pool_protocol,
    validate_replay_identity,
    validate_scene_info,
)


class FixedSceneRoboTwinEnv(RoboTwinMultiTaskEnv):
    """Replay one declared task serially; preserve exact per-episode noise seeds."""

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        if not cfg.get("is_eval") or num_envs != 1 or total_num_processes != 1:
            raise ValueError(
                "Fixed evaluation requires one eval environment and worker"
            )
        if seed_offset != 0 or len(cfg.task_names) != 1 or cfg.enable_offload:
            raise ValueError(
                "Fixed evaluation requires one resident task at offset zero"
            )
        registry = list(cfg.fixed_evaluation_tasks)
        self.fixed_pool = load_pool(
            Path(cfg.fixed_evaluation_pool), task_names=registry
        )
        import os

        validate_pool_protocol(
            self.fixed_pool,
            task_config=OmegaConf.to_container(cfg.task_config, resolve=True),
            task_horizons=dict(cfg.task_horizons),
            robotwin_root=Path(os.environ["ROBOTWIN_PATH"]),
            task_names=registry,
        )
        task = cfg.task_names[0]
        count = int(cfg.rollout_epoch)
        self.entries = self.fixed_pool["tasks"][task][:count]
        if len(self.entries) != count:
            raise ValueError("Evaluation exceeds the declared pool")
        self.record_path = Path(cfg.fixed_evaluation_records)
        if self.record_path.exists():
            raise FileExistsError(self.record_path)
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        self.episode_index = -1
        self.recorded = True
        super().__init__(
            cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics
        )

    def reset(self, *, seed=None, options=None, env_idx=None):
        if seed is not None or options is not None or env_idx is not None:
            raise ValueError("Fixed replay cannot override scene selection")
        if not self.recorded:
            raise RuntimeError("Cannot reset an incomplete fixed episode")
        index = self.episode_index + 1
        if index >= len(self.entries):
            raise RuntimeError("Fixed episode pool exhausted")
        entry = self.entries[index]
        self.set_rollout_tasks(
            [
                {
                    "slot": 0,
                    "task_id": 0,
                    "task": self.task_names[0],
                    "suite": self.task_names[0],
                    "reset_seed": entry["scene_seed"],
                    "round": index,
                }
            ]
        )
        obs, info = super().reset(evaluation_episode=entry)
        validate_replay_identity(
            entry, int(self._actual_reset_seeds[0]), obs["task_descriptions"]
        )
        self.scene_info = copy.deepcopy(self._slots[0].task.get_info())
        validate_scene_info(entry["episode_info"], self.scene_info)
        self.episode_index = index
        self.recorded = False
        return obs, info

    def chunk_step(self, chunk_actions):
        if self.episode_index < 0:
            raise RuntimeError("Reset required before stepping")
        if not torch.isfinite(torch.as_tensor(chunk_actions)).all():
            raise RuntimeError("Nonfinite policy action")
        result = super().chunk_step(chunk_actions)
        if bool((self._terminated | self._truncated)[0]) and not self.recorded:
            row = self.rollout_episode_records()[0]
            row.update(
                episode=self.episode_index,
                pool_index=self.fixed_pool["protocol"]["selection_start"]
                + self.episode_index,
                instruction=self.entries[self.episode_index]["instruction"],
                actual_scene_info=self.scene_info,
            )
            with self.record_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            self.recorded = True
        return result
