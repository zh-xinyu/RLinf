# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Frozen slots may skip inference while all live outputs and RNG stay exact."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("gr00t")
from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
    GR00T_N1_5_ForRLActionPrediction as Policy,
)
from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
    _cat_native_rollout_outputs,
)
from rlinf.models.embodiment.gr00t.gr00t_n1d5.native_rollout import (
    sample_native_rollout_noise,
)


def test_live_rows_and_rng_exact_across_frozen_slots_and_round_reset():
    class FakePolicy:
        training = False
        opd_native_rollout_batch_size = 1
        device = torch.device("cpu")

        def __init__(self, skip):
            self.skip = skip
            self.calls = 0
            self.action_head = SimpleNamespace(
                config=SimpleNamespace(action_horizon=2, action_dim=3),
                rl_config=SimpleNamespace(noise_method="flow_ode"),
                num_inference_timesteps=4,
                _opd_rollout_initial_noise_dtype=torch.float32,
                sample_noise=lambda shape, device: torch.randn(
                    shape, dtype=torch.bfloat16, device=device
                ),
            )

        def predict_action_batch(self, obs, mode="train", **kwargs):
            if self.skip and "opd_rollout_active_mask" in obs:
                return Policy._predict_active_rollout_rows(self, obs, mode, kwargs)
            outputs = []
            for row in range(len(obs["states"])):
                self.calls += 1
                noise = sample_native_rollout_noise(
                    self.action_head, 1, self.device, torch.float32
                )
                action = noise["initial"] + obs["states"][row : row + 1, None, :3]
                outputs.append(
                    (
                        action,
                        {
                            "forward_inputs": {
                                "chains": action[:, None],
                                "states": obs["states"][row : row + 1],
                            },
                            "prev_values": None,
                            "prev_logprobs": action.clone(),
                        },
                    )
                )
            return torch.cat([x[0] for x in outputs]), _cat_native_rollout_outputs(
                [x[1] for x in outputs]
            )

    masks = [
        torch.tensor(m)
        for m in [[True] * 4, [False, True, False, True], [False] * 4, [True] * 4]
    ]
    snapshots = []
    policies = [FakePolicy(False), FakePolicy(True)]
    for policy in policies:
        torch.manual_seed(782)
        steps = []
        for step, mask in enumerate(masks):
            result = policy.predict_action_batch(
                {
                    "states": torch.arange(32.0).reshape(4, 8) + step,
                    "opd_rollout_active_mask": mask,
                }
            )
            steps.append((result, torch.get_rng_state()))
        snapshots.append(steps)
    for mask, reference, optimized in zip(masks, *snapshots, strict=True):
        assert torch.equal(reference[1], optimized[1])
        assert torch.equal(reference[0][0][mask], optimized[0][0][mask])
        assert torch.equal(
            reference[0][1]["forward_inputs"]["chains"][mask],
            optimized[0][1]["forward_inputs"]["chains"][mask],
        )
    assert policies[0].calls == 16 and policies[1].calls == 10


def test_env_mask_uses_latched_final_done_and_keeps_input_unchanged():
    from rlinf.workers.env.env_worker import EnvWorker

    cfg = OmegaConf.create(
        {
            "rollout": {"model": {"opd_skip_finished_rollout": True}},
            "env": {
                "train": {
                    "freeze_finished_episodes": True,
                    "auto_reset": False,
                    "ignore_terminations": False,
                }
            },
            "algorithm": {"loss_type": "analytic_local_kl"},
        }
    )
    worker = SimpleNamespace(cfg=cfg, enable_rlt=False)
    batch = {
        "obs": {"states": torch.zeros(3, 8)},
        "final_obs": None,
        "dones": torch.tensor(
            [[False, False, True], [False, True, True], [False, False, False]]
        ),
    }
    result = EnvWorker._build_rollout_input_data(worker, batch)
    assert result["obs"]["opd_rollout_active_mask"].tolist() == [False, False, True]
    assert "opd_rollout_active_mask" not in batch["obs"]
    cfg.env.train.auto_reset = True
    with pytest.raises(ValueError, match="frozen"):
        EnvWorker._build_rollout_input_data(worker, batch)


def test_launcher_enables_skipping_without_changing_actor_or_teachers(tmp_path):
    from rlinf.utils.rollout_allocation import training_signature
    from toolkits.libero_gr00t import launch_production as launcher

    paths = {
        key: tmp_path / key
        for key in [
            "student",
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
        ]
    }
    args = SimpleNamespace(
        python=tmp_path / "runtime/bin/python",
        output=tmp_path / "out",
        gpus=[0, 1, 2, 3],
        config_name="libero_mopd_formal_m1",
        updates=200,
        rollout_steps=480,
        execution_protocol="official-mixed",
        run_purpose="formal",
    )
    base = launcher.build_config(args, paths)
    args.skip_finished_rollout = True
    optimized = launcher.build_config(args, paths)
    assert optimized.actor == base.actor
    assert optimized.rollout.expert_models == base.rollout.expert_models
    assert optimized.env == base.env
    assert (
        training_signature(optimized)["libero_protocol"]["skip_finished_rollout"]
        is True
    )
    args.native_rollout_batch_size = 4
    with pytest.raises(ValueError, match="native B1"):
        launcher.build_config(args, paths)
