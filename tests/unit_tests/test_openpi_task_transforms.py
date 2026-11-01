# Copyright 2026 The RLinf Authors.
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
"""Task coordinate routing and actual ODE refresh, without model weights."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("openpi")
from rlinf.models.embodiment.openpi.openpi_action_model import (  # noqa: E402
    OpenPi0ForRLActionPrediction as Policy,
)


def make_policy():
    policy = SimpleNamespace(
        config=SimpleNamespace(
            action_chunk=2,
            action_horizon=3,
            action_dim=1,
            action_env_dim=1,
            noise_method="flow_ode",
            num_steps=2,
            joint_logprob=False,
            is_nft=False,
            add_value_head=False,
        ),
        weight=torch.tensor(1.0),
        use_vlm_value=False,
        action_in_proj=SimpleNamespace(weight=torch.zeros(1)),
    )
    for name in (
        "setup_task_wrappers",
        "_opd_transform_ids",
        "input_transform",
        "output_transform",
        "flow_refresh_forward",
        "sample_actions",
        "_sample_actions_with_prefix_cache",
        "sample_mean_var_val",
        "_get_timesteps",
        "_get_noise_level",
        "_init_nft_state",
        "_update_nft_state",
    ):
        setattr(policy, name, MethodType(getattr(Policy, name), policy))

    def input_fn(offset):
        def transform(sample):
            return {
                "state": sample["observation/state"] + offset,
                "image": {},
                "image_mask": {},
                "tokenized_prompt": np.array([3]),
                "tokenized_prompt_mask": np.array([True]),
            }

        return transform

    def output_fn(offset):
        return lambda sample: {"actions": sample["actions"] - offset}

    policy.setup_task_wrappers(
        {
            "a": ((input_fn(10),), (output_fn(10),)),
            "b": ((input_fn(20),), (output_fn(20),)),
        }
    )
    policy.precision_processor = lambda data: data
    policy.sample_noise = lambda shape, device: torch.randn(shape, device=device)
    policy._preprocess_observation = lambda obs, train: (
        [],
        [],
        obs.tokenized_prompt,
        obs.tokenized_prompt_mask,
        obs.state,
    )
    policy._build_prefix_cache = lambda *args: (None, None, None)
    policy.get_velocity = lambda state, x, *args: (
        torch.ones_like(x) * policy.weight,
        None,
    )
    policy.get_logprob_norm = lambda sample, mu, sigma: torch.zeros_like(sample)
    return policy


def observations():
    return {
        "observation/state": torch.tensor([[1.0], [2.0], [3.0]]),
        "prompt": ["b", "a", "b"],
        "opd_task_ids": torch.tensor([1, 0, 1]),
    }


def test_mixed_task_order_and_output_inverse():
    policy = make_policy()
    obs = observations()
    transformed = policy.input_transform(obs, transpose=False)
    torch.testing.assert_close(
        transformed["state"][:, 0], torch.tensor([21.0, 12.0, 23.0])
    )
    output = policy.output_transform(
        {"actions": transformed["state"][:, None].expand(-1, 3, -1)},
        obs["opd_task_ids"],
    )
    assert output["actions"].shape == (3, 2, 1)
    torch.testing.assert_close(output["actions"][:, 0], obs["observation/state"])
    # Replay uses original tokens and raw observations, preserving task coordinates.
    replay = {key: value for key, value in obs.items() if key != "prompt"}
    replay["tokenized_prompt"] = torch.tensor([[8], [9], [10]])
    replay["tokenized_prompt_mask"] = torch.ones(3, 1, dtype=torch.bool)
    replayed = policy.input_transform(replay, transpose=False)
    torch.testing.assert_close(replayed["state"], transformed["state"])
    assert replayed["tokenized_prompt"] is replay["tokenized_prompt"]


@pytest.mark.parametrize("ids", [None, [0, 1], [0.0, 1.0, 0.0], [0, -1, 1], [0, 2, 1]])
def test_missing_or_invalid_task_ids_fail(ids):
    policy = make_policy()
    obs = observations()
    obs["opd_task_ids"] = ids
    with pytest.raises(ValueError):
        policy.input_transform(obs, transpose=False)
    with pytest.raises(ValueError):
        policy.output_transform({"actions": torch.zeros(3, 3, 1)}, ids)


def test_refresh_uses_fresh_noise_and_current_student_velocity():
    policy = make_policy()
    obs = observations()
    obs["chains"] = torch.full((3, 3, 3, 1), 999.0)
    torch.manual_seed(17)
    first = policy.flow_refresh_forward(obs)["chains"]
    second = policy.flow_refresh_forward(obs)["chains"]
    assert first.shape == (3, 3, 3, 1)
    assert not torch.equal(first[:, 0], second[:, 0])
    assert not first.requires_grad
    torch.testing.assert_close(first[:, -1], first[:, 0] - 1)
    policy.weight.fill_(2)
    torch.manual_seed(17)
    current = policy.flow_refresh_forward(obs)["chains"]
    torch.testing.assert_close(current[:, 0], first[:, 0])
    torch.testing.assert_close(current[:, -1], first[:, -1] - 1)
    assert obs["chains"].eq(999).all()


def test_refresh_rejects_sde_configuration():
    policy = make_policy()
    policy.config.noise_method = "flow_sde"
    with pytest.raises(ValueError, match="flow_ode"):
        policy.flow_refresh_forward(observations())
