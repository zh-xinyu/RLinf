# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Thin instrumentation around real train workers; no simulator/model reimplementation."""

import os
from pathlib import Path

import torch

from rlinf.utils.nested_dict_process import clone_nested_to_cpu
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
from toolkits.libero_gr00t.teacher_reference import SUITES
from toolkits.libero_gr00t.teacher_training_probe import rank_config


class TeacherProbeEnvWorker(EnvWorker):
    """Use inherited quota resets, action preparation, and train interaction."""

    def __init__(self, cfg):
        super().__init__(rank_config(cfg, int(os.environ["RANK"])))
        self._action_calls = 0
        self._action_max_error = 0.0

    def env_interact_step(self, chunk_actions, stage_id):
        """Check the commands produced by the actual parent's prepare_actions call."""
        raw = (
            chunk_actions["raw_actions"]
            if isinstance(chunk_actions, dict)
            else chunk_actions
        )
        before = torch.as_tensor(raw).detach().cpu().clone()
        result = super().env_interact_step(chunk_actions, stage_id)
        prepared = torch.as_tensor(result[2]["chunk_actions"]).detach().cpu()
        if before.shape != (1, 8, 7) or prepared.shape != before.shape:
            raise ValueError("Unexpected training action shape")
        if not torch.isfinite(prepared).all():
            raise ValueError("Nonfinite physical training command")
        error = float((before - prepared).abs().max())
        if error != 0:
            raise ValueError(
                "GR00T LIBERO prepare_actions changed decoded controller commands"
            )
        self._action_calls += 1
        self._action_max_error = max(self._action_max_error, error)
        return result

    def get_probe_action_audit(self) -> dict:
        """Return cumulative evidence from actual train action preparation calls."""
        return {
            "rank": self._rank,
            "suite": SUITES[self._rank],
            "prepare_actions_calls": self._action_calls,
            "max_decoded_to_prepared_error": self._action_max_error,
            "path": "EnvWorker.env_interact_step -> prepare_actions -> LiberoEnv.chunk_step",
        }


class TeacherProbeRolloutWorker(MultiStepRolloutWorker):
    """Run one native specialist through inherited HF train generation."""

    def __init__(self, cfg):
        super().__init__(rank_config(cfg, int(os.environ["RANK"])))
        self._probe_call = 0
        self._probe_identity = None
        self._probe_captures = []

    def init_worker(self):
        """Cap this rollout process's allocator before loading the native teacher."""
        total = torch.cuda.get_device_properties(self.device).total_memory
        cap = float(self.cfg.teacher_training_probe.memory_gib) * 2**30
        torch.cuda.set_per_process_memory_fraction(min(cap / total, 1), self.device)
        super().init_worker()
        self.hf_model.requires_grad_(False)
        if (
            self.hf_model.valid_action_dim != 7
            or self.hf_model.opd_fixed_text_padding is not True
        ):
            raise ValueError(
                "Native specialist must retain seven actions and fixed570 prompts"
            )

    def set_probe_episode(self, task: int, trial: int) -> None:
        """Match the reference's explicit per-task/trial torch noise seed."""
        probe = self.cfg.teacher_training_probe
        seed = int(probe.seed) + task * 100 + trial
        torch.manual_seed(seed)
        self._probe_call = 0
        self._probe_captures = []
        self._probe_identity = {
            "suite": SUITES[self._rank],
            "task_id": task,
            "trial_id": trial,
            "policy_seed": seed,
            "reset_seed": int(probe.seed),
            "observation_source": "EnvWorker.env.train",
        }

    def predict(self, env_obs, mode="train"):
        """Audit real HF train predictions and optionally save exact noninitial inputs."""
        if mode != "train" or self._probe_identity is None:
            raise ValueError(
                "Diagnostic must use the initialized production train path"
            )
        if list(env_obs["task_groups"]) != [SUITES[self._rank]]:
            raise ValueError("Native teacher received another suite's observation")
        capture = self._probe_call in self.cfg.teacher_training_probe.capture_chunks
        saved_obs = clone_nested_to_cpu(env_obs) if capture else None
        actions, result = super().predict(env_obs, mode=mode)
        inputs = result["forward_inputs"]
        if (
            tuple(actions.shape) != (1, 8, 7)
            or tuple(inputs["chains"].shape) != (1, 5, 16, 32)
            or int(inputs["state_mask"].sum()) != 8
            or inputs["eagle_input_ids"].shape[-1] != 570
            or not bool(torch.all(inputs["embodiment_id"] == 31))
            or not bool(torch.isfinite(actions).all())
            or not bool(torch.all(actions[..., -1].abs() == 1))
        ):
            raise ValueError(
                "Native teacher state/projector/prompt/chain/action contract changed"
            )
        if capture:
            identity = dict(self._probe_identity, chunk_index=self._probe_call)
            name = f"{identity['suite']}_task{identity['task_id']}_trial{identity['trial_id']}_chunk{self._probe_call}.pt"
            relative = Path("captures") / name
            path = Path(self.cfg.teacher_training_probe.output) / relative
            path.parent.mkdir(exist_ok=True)
            keys = (
                "chains",
                "opd_raw_states",
                "state",
                "state_mask",
                "embodiment_id",
                "eagle_input_ids",
                "eagle_attention_mask",
                "eagle_pixel_values",
                "eagle_image_sizes",
            )
            with path.open("xb") as stream:
                torch.save(
                    {
                        "identity": identity,
                        "env_obs": saved_obs,
                        "forward_inputs": clone_nested_to_cpu(
                            {key: inputs[key] for key in keys}
                        ),
                        "actions": clone_nested_to_cpu(actions),
                    },
                    stream,
                )
            self._probe_captures.append({"path": str(relative), **identity})
        self._probe_call += 1
        return actions, result

    def get_probe_audit(self) -> dict:
        """Report actual train calls, native input checks, captures, and allocator peak."""
        if self._probe_call != self.n_train_chunk_steps + 1:
            raise ValueError(
                "Did not traverse all production train/bootstrap inference calls"
            )
        return {
            "rank": self._rank,
            **self._probe_identity,
            "train_inference_calls": self._probe_call,
            "contract": {
                "state_valid_dim": 8,
                "projector": 31,
                "text_length": 570,
                "generated_horizon": 16,
                "executed_horizon": 8,
                "valid_action_dim": 7,
                "denoise_steps": 4,
                "mode": "train",
                "noise_method": "flow_ode",
            },
            "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(self.device) / 2**30,
            "captures": self._probe_captures,
        }
