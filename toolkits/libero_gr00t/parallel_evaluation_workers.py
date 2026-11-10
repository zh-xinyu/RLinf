# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Bounded evaluation adapters for the ordinary LIBERO environment/rollout workers."""

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
from toolkits.libero_gr00t.parallel_evaluation import EpisodePolicy, write_json


class EvaluationPackets:
    """Delegate physics to LiberoEnv and stage exact initial states on reset."""

    def __init__(self, env: Any, cfg: DictConfig) -> None:
        self.env = env
        self.cfg = cfg
        self.packet = 0
        self.rows = []
        self.recorded = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def is_start(self) -> bool:
        return self.env.is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self.env.is_start = value

    def reset(self) -> tuple[dict, dict]:
        begin = self.packet * self.env.num_envs
        trials = list(self.cfg.trials[begin : begin + self.env.num_envs])
        self.env.set_evaluation_episodes(int(self.cfg.task_id), trials)
        self.keys = [(int(self.cfg.task_id), trial) for trial in trials]
        self.packet += 1
        self.recorded = False
        return self.env.reset()

    def chunk_step(self, actions: Any) -> tuple:
        output = self.env.chunk_step(actions)
        if not self.recorded and bool((output[2] | output[3])[:, -1].all()):
            self.rows.extend(
                dict(row, settling_steps=int(self.env.cfg.reset_settle_steps))
                for row in self.env.rollout_episode_records()
            )
            self.recorded = True
            write_json(Path(self.cfg.output) / "episodes.json", self.rows)
        return output


class ParallelEvalEnvWorker(EnvWorker):
    """Use standard reset/step/channels with bounded fixed-state packets."""

    def init_worker(self) -> None:
        super().init_worker()
        if self.stage_num != 1 or self._world_size != 1:
            raise ValueError("Dispatch one task per isolated single-rank worker pair")
        self.eval_env_list = [
            EvaluationPackets(self.eval_env_list[0], self.cfg.evaluation)
        ]
        self.n_eval_chunk_steps = int(self.cfg.evaluation.decisions)

    def _build_rollout_input_data(self, env_batch: dict) -> dict:
        # Evaluation-only: never invoke training distillation checks or labels.
        env = self.eval_env_list[0]
        obs = dict(env_batch["obs"])
        obs["opd_eval_keys"] = torch.tensor(env.keys, dtype=torch.long)
        obs["opd_eval_active"] = torch.as_tensor(
            ~(env._episode_terminated | env._episode_truncated), dtype=torch.bool
        )
        return {"obs": obs, "final_obs": env_batch.get("final_obs")}

    def finish_rollout(self, mode: str = "train") -> None:
        if mode != "eval" or not self.eval_env_list[0].recorded:
            raise RuntimeError("Evaluation packet did not finish at native horizons")
        # Explicit packets replace ordered-pool iteration in the generic worker.


class ParallelEvalRolloutWorker(MultiStepRolloutWorker):
    """Reuse checkpoint construction/inference; isolate episode noise streams."""

    def init_worker(self) -> None:
        import os

        from omegaconf import open_dict

        torch.set_num_threads(1)
        total = torch.cuda.mem_get_info()[1]
        torch.cuda.set_per_process_memory_fraction(
            float(os.environ.get("OPD_EVAL_MEMORY_GIB", "16")) * 2**30 / total
        )
        weights = self.cfg.runner.ckpt_path
        # Let the standard worker construct exactly the training policy, then
        # validate full weights on CPU before any dtype-coercing copy can occur.
        with open_dict(self.cfg):
            self.cfg.runner.ckpt_path = None
        try:
            super().init_worker()
        finally:
            with open_dict(self.cfg):
                self.cfg.runner.ckpt_path = weights
        if weights:
            from toolkits.libero_gr00t.evaluate_full_checkpoint import load_full_weights

            validation = load_full_weights(self.hf_model.cpu(), Path(weights))
            self.hf_model.to(self.device)
            write_json(Path(self.cfg.evaluation.output) / "reload.json", validation)
        self.hf_model.requires_grad_(False)
        self.n_eval_chunk_steps = int(self.cfg.evaluation.decisions)
        self.episode_policy = EpisodePolicy(
            int(self.cfg.evaluation.seed), bool(self.cfg.evaluation.trace)
        )

    def _predict_rollout_actions(
        self, env_obs: dict, mode: str = "train", **kwargs
    ) -> tuple[torch.Tensor, dict]:
        if mode != "eval":
            raise ValueError("Independent evaluation worker cannot train")
        obs = dict(env_obs)
        keys = obs.pop("opd_eval_keys").tolist()
        active = obs.pop("opd_eval_active").tolist()
        actions = self.episode_policy.predict(self.hf_model, obs, keys, active)
        if self.episode_policy.trace is not None:
            write_json(
                Path(self.cfg.evaluation.output) / "action_trace.json",
                self.episode_policy.trace,
            )
        return actions, {}
