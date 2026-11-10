# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Native rollout coverage, input shapes, noise consumption and environment order."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("gr00t")
from rlinf.models.embodiment.gr00t.gr00t_n1d5.native_rollout import (
    native_rollout_plan,
    pack_native_rollout_inputs,
    sample_native_rollout_noise,
)


def prepared(length: int, images: int = 2) -> dict:
    return {
        "state": torch.zeros(1, 1, 8),
        "eagle_input_ids": torch.arange(length)[None],
        "eagle_attention_mask": torch.ones(1, length, dtype=torch.long),
        "eagle_pixel_values": torch.zeros(images, 3, 4, 4),
    }


def test_plan_keeps_exact_coverage_and_image_shapes_without_duplicate_padding():
    rows = [
        prepared(3),
        prepared(5),
        prepared(3),
        prepared(3),
        prepared(5),
        prepared(3, 1),
    ]
    plan = native_rollout_plan(rows, 2)
    assert plan == [[0, 2], [3], [1, 4], [5]]
    assert sorted(i for group in plan for i in group) == list(range(6))
    batch = pack_native_rollout_inputs(rows, plan[0])
    assert batch["eagle_input_ids"].shape == (2, 3)
    assert batch["eagle_pixel_values"].shape == (4, 3, 4, 4)
    assert rows[0]["eagle_pixel_values"].shape[0] == 2
    rows[0]["eagle_attention_mask"][0, 0] = 0
    with pytest.raises(ValueError, match="without text padding"):
        native_rollout_plan(rows, 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_noise_stream_exactly_matches_individual_ode_sampling(dtype):
    class Head:
        rl_config = SimpleNamespace(noise_method="flow_ode")
        config = SimpleNamespace(action_horizon=16, action_dim=32)
        num_inference_timesteps = 4

        def sample_noise(self, shape, device):
            return torch.normal(
                0.0, 1.0, size=shape, device=device, dtype=torch.bfloat16
            )

    head = Head()
    torch.manual_seed(827)
    expected = []
    for _ in range(7):
        initial = torch.randn(1, 16, 32, dtype=dtype)
        steps = torch.stack(
            [head.sample_noise((1, 16, 32), "cpu") for _ in range(4)], 1
        )
        expected.append((initial, steps))
    end = torch.get_rng_state()
    torch.manual_seed(827)
    noise = sample_native_rollout_noise(head, 7, torch.device("cpu"), dtype)
    assert torch.equal(torch.get_rng_state(), end)
    assert torch.equal(noise["initial"], torch.cat([row[0] for row in expected]))
    assert torch.equal(noise["transitions"], torch.cat([row[1] for row in expected]))
    head.rl_config = SimpleNamespace(noise_method="flow_sde")
    with pytest.raises(ValueError, match="flow_ode"):
        sample_native_rollout_noise(head, 7, torch.device("cpu"), dtype)


def test_configuration_defaults_and_unsupported_policy_rejection():
    pytest.importorskip("gr00t")
    from rlinf.models.embodiment.gr00t.gr00t_n1d5 import (
        validate_native_rollout_batch_size,
    )

    assert validate_native_rollout_batch_size(OmegaConf.create({})) == 1
    cfg = OmegaConf.create(
        {
            "opd_fixed_text_padding": False,
            "opd_native_rollout_batch_size": 4,
            "rl_head_config": {"noise_method": "flow_ode", "disable_dropout": True},
        }
    )
    assert validate_native_rollout_batch_size(cfg) == 4
    for value in [True, 0, -1, 1.5]:
        cfg.opd_native_rollout_batch_size = value
        with pytest.raises(ValueError, match="positive integer"):
            validate_native_rollout_batch_size(cfg)
    cfg.opd_native_rollout_batch_size = 4
    cfg.opd_fixed_text_padding = True
    with pytest.raises(ValueError, match="native text"):
        validate_native_rollout_batch_size(cfg)


@pytest.mark.parametrize("limit", [2, 4])
@pytest.mark.parametrize("noise_dtype", [torch.bfloat16, torch.float32])
def test_real_policy_path_restores_actions_flow_payload_and_rng(
    monkeypatch, limit, noise_dtype
):
    pytest.importorskip("gr00t")
    from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
        GR00T_N1_5_ForRLActionPrediction as Policy,
    )
    from rlinf.models.embodiment.gr00t.simulation_io import OBS_CONVERSION

    class FakePolicy:
        predict_action_batch = Policy.predict_action_batch
        _check_state_is_batched = Policy._check_state_is_batched
        obs_convert_fn = staticmethod(OBS_CONVERSION["libero_grouped"])
        opd_fixed_text_padding = False
        _opd_state_adapter_validated = True
        output_action_chunks = 2
        device = torch.device("cpu")
        inference_autocast_dtype = torch.bfloat16
        training = False

        def __init__(self, batch_size):
            self.opd_native_rollout_batch_size = batch_size
            self.calls = []
            self.action_head = SimpleNamespace(
                rl_config=SimpleNamespace(noise_method="flow_ode"),
                config=SimpleNamespace(action_horizon=2, action_dim=7),
                num_inference_timesteps=4,
                sample_noise=lambda shape, device: torch.randn(
                    shape, dtype=torch.bfloat16, device=device
                ),
            )

        def apply_transforms(self, obs):
            assert obs["state.eef_pos_absolute"].shape[0] == 1
            length = len(obs["annotation.human.action.task_description"][0])
            data = prepared(length)
            data.update(
                state=torch.as_tensor(obs["state.eef_pos_absolute"]),
                state_mask=torch.ones(1, 1, 3, dtype=torch.bool),
                eagle_image_sizes=torch.ones(2, 2),
                embodiment_id=torch.tensor([31]),
            )
            return data

        def _get_rl_action(self, inputs, mode, rollout_noise=None):
            size = inputs["state"].shape[0]
            self.calls.append(size)
            if callable(rollout_noise):
                rollout_noise = rollout_noise(self.device, noise_dtype)
            if rollout_noise is None:
                rollout_noise = sample_native_rollout_noise(
                    self.action_head, size, self.device, noise_dtype
                )
            action = (
                rollout_noise["initial"].float() + inputs["state"][:, :, :1].float()
            )
            stored = dict(inputs)
            stored["eagle_pixel_values"] = inputs["eagle_pixel_values"].reshape(
                size, 2, 3, 4, 4
            )
            stored["eagle_image_sizes"] = inputs["eagle_image_sizes"].reshape(
                size, 2, 2
            )
            stored.update(
                chains=action[:, None].repeat(1, 5, 1, 1),
                denoise_inds=torch.ones(size, 4, dtype=torch.long),
            )
            return action, {
                "forward_inputs": stored,
                "prev_values": None,
                "prev_logprobs": action.clone(),
            }

        def _get_unnormalized_action(self, actions):
            return {"action": actions.numpy()}

        @staticmethod
        def action_convert_fn(actions, chunk_size):
            return actions["action"][:, :chunk_size]

    obs = {
        "states": torch.arange(56).reshape(7, 8).float() / 10,
        "main_images": torch.zeros(7, 4, 4, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(7, 4, 4, 3, dtype=torch.uint8),
        "task_descriptions": ["aa", "bbbb", "cc", "dd", "eeee", "ff", "gggg"],
    }
    original = FakePolicy(1)
    torch.manual_seed(19)
    expected = original.predict_action_batch(dict(obs))
    state = torch.get_rng_state()
    optimized = FakePolicy(limit)
    torch.manual_seed(19)
    actual = optimized.predict_action_batch(dict(obs))
    assert torch.equal(torch.get_rng_state(), state)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    for key in actual[1]["forward_inputs"]:
        torch.testing.assert_close(
            actual[1]["forward_inputs"][key],
            expected[1]["forward_inputs"][key],
            rtol=0,
            atol=0,
        )
    assert any(size > 1 for size in optimized.calls)
    assert sum(optimized.calls) == len(obs["task_descriptions"])
    optimized.training = True
    with pytest.raises(ValueError, match="model.eval"):
        optimized.predict_action_batch(dict(obs))


def test_launch_keeps_actor_and_teacher_contracts_unchanged(tmp_path):
    from rlinf.utils.rollout_allocation import training_signature
    from toolkits.libero_gr00t import launch_production as production

    paths = {
        k: tmp_path / k
        for k in [
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
    original = production.build_config(args, paths)
    args.native_rollout_batch_size = 4
    optimized = production.build_config(args, paths)
    assert optimized.actor == original.actor
    assert optimized.rollout.expert_models == original.rollout.expert_models
    assert optimized.rollout.model.opd_native_rollout_batch_size == 4
    assert (
        training_signature(optimized)["libero_protocol"]["native_rollout_batch_size"]
        == 4
    )
    assert (
        "native_rollout_batch_size"
        not in training_signature(original)["libero_protocol"]
    )


def test_rollout_worker_applies_rollout_only_batch_limit(monkeypatch):
    from rlinf.workers.rollout.hf import huggingface_worker as module

    class StopBeforeModel(Exception):
        pass

    captured = {}

    def fake_model(cfg):
        captured.update(OmegaConf.to_container(cfg))
        raise StopBeforeModel

    monkeypatch.setattr(module, "get_model", fake_model)
    actor = OmegaConf.create({"model_path": "student", "precision": "bf16"})
    worker = SimpleNamespace(
        model_cfg=actor,
        cfg=OmegaConf.create(
            {
                "rollout": {
                    "model": {
                        "model_path": "student",
                        "precision": "bf16",
                        "opd_native_rollout_batch_size": 4,
                    }
                }
            }
        ),
    )
    with pytest.raises(StopBeforeModel):
        module.MultiStepRolloutWorker.init_worker(worker)
    assert captured["opd_native_rollout_batch_size"] == 4
    assert "opd_native_rollout_batch_size" not in actor


def test_backbone_batching_keeps_head_b1_and_noise_row_correspondence():
    from transformers.feature_extraction_utils import BatchFeature

    from rlinf.models.embodiment.gr00t.gr00t_n1d5.native_rollout import native_head_rows

    calls = []

    class Head:
        def get_rl_action(self, backbone, actions, mode, rollout_noise):
            calls.append(backbone.backbone_features.shape[0])
            noise = rollout_noise(torch.device("cpu"), torch.float32)
            value = backbone.backbone_features[:, :1] + noise["initial"]
            return BatchFeature(data={"action_pred": value}), {
                "actions": value,
                "chains": value[:, None],
            }

    features = torch.arange(6.0).reshape(3, 2)
    initial = torch.arange(3.0).reshape(3, 1)
    prediction, result = native_head_rows(
        Head(),
        BatchFeature(data={"backbone_features": features}),
        BatchFeature(data={"state": features}),
        "train",
        lambda device, dtype: {"initial": initial},
    )
    assert calls == [1, 1, 1]
    torch.testing.assert_close(prediction.action_pred, features[:, :1] + initial)
    torch.testing.assert_close(result["actions"], prediction.action_pred)
