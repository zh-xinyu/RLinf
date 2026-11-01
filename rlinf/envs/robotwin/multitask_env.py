# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task-routed RoboTwin rollouts using the benchmark's native policy execution.

Each slot owns one episode, stops at success or its task-native horizon, and
retains its terminal observation until the next allocation/reset. No planner
expert trajectories or success-filtered seeds are used.
"""

import copy
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

# Pinned RoboTwin task_config/_eval_step_limit.yml (RLinf_support dca9ec6).
TASK_HORIZONS = {
    "adjust_bottle": 400,
    "handover_mic": 600,
    "place_container_plate": 400,
    "stack_bowls_two": 900,
}


class NativeTaskSlot:
    """One task with native take_action execution and current-episode language."""

    def __init__(self, task_name, horizon, task_config, instruction_type="seen"):
        from robotwin.envs.vector_env import VectorEnv

        class ConfigOnlyVectorEnv(VectorEnv):
            def _init_envs(self):
                pass

        config = copy.deepcopy(task_config)
        config.update(task_name=task_name, step_lim=horizon)
        resolver = ConfigOnlyVectorEnv(config, n_envs=1)
        self.args = resolver.args
        resolver.env_thread_pool.shutdown()
        self.task_name = task_name
        self.horizon = horizon
        self.instruction_type = instruction_type
        self.task = None
        self.actual_seed = None

    def reset(self, seed, *, instruction=None, strict_seed=False):
        from description.utils.generate_episode_instructions import (
            generate_episode_descriptions,
        )
        from envs.utils.create_actor import UnStableError
        from robotwin.envs.vector_env import class_decorator

        self.close()
        # Retry only physically unstable resets, never planner/task failures.
        # Record the actual seed so these benchmark-native retries are visible.
        for attempt in range(1 if strict_seed else 20):
            actual_seed = seed + attempt
            self.task = class_decorator(self.task_name)
            try:
                self.task.setup_demo(
                    now_ep_num=actual_seed, seed=actual_seed, **self.args
                )
                break
            except UnStableError as error:
                self.close()
                if strict_seed:
                    raise RuntimeError(
                        f"Fixed evaluation seed {seed} is unstable; refusing reseed"
                    ) from error
        else:
            raise RuntimeError(f"20 unstable resets for {self.task_name}, seed={seed}")
        self.task.step_lim = self.horizon
        self.task.run_steps = 0
        self.task.reward_step = 0
        self.actual_seed = actual_seed
        if instruction is None:
            descriptions = generate_episode_descriptions(
                self.task_name, [self.task.get_info()], 1, actual_seed
            )[0][self.instruction_type]
            instruction = str(np.random.default_rng(actual_seed).choice(descriptions))
        self.task.set_instruction(instruction)
        self.last_obs = self.observe()
        return self.last_obs

    def observe(self):
        from robotwin.envs.vector_env import update_obs

        obs = update_obs(self.task.get_obs())
        obs["instruction"] = self.task.get_instruction()
        return obs

    def step_chunk(self, actions):
        """Return exact control-target count; never execute padded actions."""
        executed = 0
        for action in actions:
            if self.task.eval_success or self.task.take_action_cnt >= self.horizon:
                break
            self.task.take_action(action, action_type="qpos")
            executed += 1
        if executed:
            self.last_obs = self.observe()
        success = bool(self.task.eval_success)
        truncated = self.task.take_action_cnt >= self.horizon and not success
        return self.last_obs, executed, success, truncated

    def close(self):
        if self.task is not None:
            self.task.close_env(clear_cache=True)
            self.task = None


class RoboTwinMultiTaskEnv(RoboTwinEnv):
    """Fixed-size multi-task pool compatible with suite-quota allocation."""

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        # Do not construct the single-task parent backend.
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.seed = int(cfg.seed) + seed_offset
        self.base_seed = int(cfg.seed)
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.record_metrics = record_metrics
        self.auto_reset = bool(cfg.auto_reset)
        self.ignore_terminations = bool(cfg.ignore_terminations)
        if self.auto_reset or self.ignore_terminations:
            raise ValueError("Multi-task quota rollouts require frozen terminal slots")
        self.group_size = int(cfg.group_size)
        if self.group_size != 1:
            raise ValueError("Multi-task RoboTwin requires group_size=1")
        self.num_group = num_envs
        self.use_fixed_reset_state_ids = cfg.get("use_fixed_reset_state_ids", False)
        self.video_cfg = cfg.video_cfg
        self.center_crop = bool(cfg.get("center_crop", False))
        self.episode_horizon_source = "task_horizon"
        self.task_names = list(cfg.task_names)
        if not self.task_names or len(set(self.task_names)) != len(self.task_names):
            raise ValueError("task_names must be unique and nonempty")
        horizons = dict(cfg.get("task_horizons", TASK_HORIZONS))
        self.task_specs = [
            {
                "task_name": name,
                "horizon": int(horizons[name]),
                "metadata_view": {"task_soup": name},
            }
            for name in self.task_names
        ]
        self.horizon = max(spec["horizon"] for spec in self.task_specs)
        if int(cfg.max_steps_per_rollout_epoch) < self.horizon:
            raise ValueError("Rollout length must cover every task-native horizon")
        self._is_start = True
        self._reset_round = 0
        self._allocation = None
        self._slots = [None] * num_envs
        self._slot_task_ids = [i % len(self.task_names) for i in range(num_envs)]
        self._raw_obs = [None] * num_envs
        self._actual_reset_seeds = [None] * num_envs
        self._pool = ThreadPoolExecutor(max_workers=num_envs)
        self._elapsed_steps = torch.zeros(num_envs, dtype=torch.long)
        self.success_once = torch.zeros(num_envs, dtype=torch.bool)
        self._terminated = torch.zeros(num_envs, dtype=torch.bool)
        self._truncated = torch.zeros(num_envs, dtype=torch.bool)
        self.returns = torch.zeros(num_envs)
        self.update_reset_state_ids()

    @property
    def device(self):
        # Environment bookkeeping/observation transport stays on CPU.
        return torch.device("cpu")

    def _make_slot(self, task_id):
        os.environ["ASSETS_PATH"] = str(self.cfg.assets_path)
        spec = self.task_specs[task_id]
        return NativeTaskSlot(
            spec["task_name"],
            spec["horizon"],
            OmegaConf.to_container(self.cfg.task_config, resolve=True),
            self.cfg.get("instruction_type", "seen"),
        )

    def set_rollout_tasks(self, rows):
        """Assign one task/seed per slot before reset; preserve plan metadata."""
        rows = sorted((dict(row) for row in rows), key=lambda row: row["slot"])
        if [row["slot"] for row in rows] != list(range(self.num_envs)):
            raise ValueError("Allocation must assign every local slot exactly once")
        for row in rows:
            task_id = int(row["task_id"])
            if not 0 <= task_id < len(self.task_names):
                raise ValueError(f"Invalid task_id: {task_id}")
            name = self.task_names[task_id]
            if row["task"] != name or row["suite"] != name:
                raise ValueError("Allocation task/suite disagree with task_id")
        for row in rows:
            slot = row["slot"]
            task_id = int(row["task_id"])
            if self._slot_task_ids[slot] != task_id and self._slots[slot] is not None:
                self._slots[slot].close()
                self._slots[slot] = None
            self._slot_task_ids[slot] = task_id
        self._allocation = rows
        self.reset_state_ids = torch.tensor([r["reset_seed"] for r in rows])

    def update_reset_state_ids(self, env_idx=None):
        if self._allocation is not None:
            return
        if self.use_fixed_reset_state_ids and hasattr(self, "reset_state_ids"):
            return
        # Worker seed_offset partitions otherwise identical deterministic streams.
        rng = np.random.default_rng(self.seed + self._reset_round)
        self.reset_state_ids = torch.from_numpy(
            rng.integers(0, 2**31 - 1, size=self.num_envs)
        )
        self._reset_round += 1

    def _observations(self):
        obs = self._extract_obs_image(self._raw_obs)
        obs["task_groups"] = [self.task_names[i] for i in self._slot_task_ids]
        obs["extra_view_images"] = None
        return obs

    def reset(self, *, seed=None, options=None, env_idx=None, evaluation_episode=None):
        if env_idx is not None:
            raise ValueError("Quota rollouts reset the complete pool between rounds")
        if seed is not None:
            self.reset_state_ids = torch.arange(self.num_envs) + int(seed)
        if evaluation_episode is not None:
            if self.num_envs != 1:
                raise ValueError("Fixed evaluation episodes require one environment")
            if int(self.reset_state_ids[0]) != int(evaluation_episode["scene_seed"]):
                raise ValueError("Fixed evaluation seed disagrees with allocation")
            if (
                not isinstance(evaluation_episode["instruction"], str)
                or not evaluation_episode["instruction"].strip()
            ):
                raise ValueError("Fixed evaluation instruction must be nonempty")
        # setup_demo changes global random generators; do not reset in threads.
        for i, task_id in enumerate(self._slot_task_ids):
            if self._slots[i] is None:
                self._slots[i] = self._make_slot(task_id)
            reset_kwargs = (
                {}
                if evaluation_episode is None
                else {
                    "instruction": evaluation_episode["instruction"],
                    "strict_seed": True,
                }
            )
            self._raw_obs[i] = self._slots[i].reset(
                int(self.reset_state_ids[i]), **reset_kwargs
            )
            self._actual_reset_seeds[i] = self._slots[i].actual_seed
        self._elapsed_steps.zero_()
        self.success_once.zero_()
        self._terminated.zero_()
        self._truncated.zero_()
        self.returns.zero_()
        self._is_start = False
        return self._observations(), {}

    def chunk_step(self, chunk_actions):
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu().numpy()
        actions = np.asarray(chunk_actions)
        if (
            actions.ndim != 3
            or actions.shape[0] != self.num_envs
            or actions.shape[2] != 14
        ):
            raise ValueError("Expected actions [num_envs, chunk_length, 14]")
        length = actions.shape[1]
        rewards = torch.zeros(self.num_envs, length)
        terminated = self._terminated[:, None].expand(-1, length).clone()
        truncated = self._truncated[:, None].expand(-1, length).clone()
        active = (~(self._terminated | self._truncated)).nonzero().flatten().tolist()
        futures = {
            i: self._pool.submit(self._slots[i].step_chunk, actions[i]) for i in active
        }
        for i, future in futures.items():
            obs, executed, success, timeout = future.result(timeout=1200)
            if not 0 < executed <= length:
                raise RuntimeError(f"Active slot {i} executed invalid count {executed}")
            self._raw_obs[i] = obs
            self._elapsed_steps[i] += executed
            self.success_once[i] |= success
            if success:
                rewards[i, executed - 1] = float(self.cfg.reward_coef)
                terminated[i, executed - 1 :] = True
                self._terminated[i] = True
            if timeout:
                truncated[i, executed - 1 :] = True
                self._truncated[i] = True
        self.returns += rewards.sum(dim=1)
        info = {
            "episode": {
                "success_once": self.success_once.clone(),
                "return": self.returns.clone(),
                "episode_len": self._elapsed_steps.clone(),
                "reward": self.returns / self._elapsed_steps.clamp_min(1),
            }
        }
        return [self._observations()], rewards, terminated, truncated, [info]

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.chunk_step(actions[:, None])
        return obs[-1], rewards[:, 0], terminated[:, 0], truncated[:, 0], infos[-1]

    def rollout_episode_records(self):
        """Count executed control targets, excluding terminal padding.

        simulation_steps follows RLinf control-step budgeting, not SAPIEN's
        variable-count physics integration substeps inside take_action.
        """
        if self._allocation is None:
            return []
        return [
            dict(
                row,
                success_once=bool(self.success_once[i]),
                simulation_steps=int(self._elapsed_steps[i]),
                actual_reset_seed=int(self._actual_reset_seeds[i]),
            )
            for i, row in enumerate(self._allocation)
        ]

    def offload(self, clear_cache=True):
        for i, slot in enumerate(self._slots):
            if slot is not None:
                slot.close()
                self._slots[i] = None

    def close(self):
        self.offload()
        self._pool.shutdown(wait=True)
