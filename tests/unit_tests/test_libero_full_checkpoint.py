# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Production full-weight artifacts must not silently become partial/low-precision loads."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from toolkits.libero_gr00t.evaluate_full_checkpoint import (
    candidate_model_config,
    checkpoint_step,
    load_full_weights,
    parse_args,
    prepare_evaluation_weights,
)


def policy():
    model = torch.nn.ModuleDict(
        {
            "backbone": torch.nn.Linear(2, 2).to(torch.bfloat16),
            "action_head": torch.nn.Linear(2, 1),
        }
    )
    return model


def test_exact_full_state_restores_both_frozen_backbone_and_fp32_head(tmp_path):
    source, target = policy(), policy()
    source["backbone"].load_state_dict(target["backbone"].state_dict())
    path = tmp_path / "full_weights.pt"
    torch.save(source.state_dict(), path)
    result = load_full_weights(target, path)
    assert result["strict"] and result["state_tensor_count"] == 4
    for key, tensor in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], tensor, rtol=0, atol=0)
    assert target["action_head"].weight.dtype == torch.float32
    assert result["base_comparison"]["unchanged_backbone_tensors"] == 2
    assert result["base_comparison"]["changed_action_head_tensors"] == 2


@pytest.mark.parametrize(
    "fault", ["partial", "nested", "shape", "dtype", "nonfinite", "backbone"]
)
def test_invalid_checkpoint_fails_before_mutating_model(tmp_path, fault):
    target = policy()
    before = {key: value.clone() for key, value in target.state_dict().items()}
    state = {key: value.clone() for key, value in policy().state_dict().items()}
    state.update(
        {
            key: value.clone()
            for key, value in before.items()
            if key.startswith("backbone.")
        }
    )
    if fault == "partial":
        del state["backbone.weight"]
    elif fault == "nested":
        state = {"model": state}
    elif fault == "shape":
        state["action_head.weight"] = torch.zeros(1, 3)
    elif fault == "dtype":
        state["action_head.weight"] = state["action_head.weight"].bfloat16()
    elif fault == "nonfinite":
        state["action_head.weight"][0, 0] = float("nan")
    else:
        state["backbone.weight"][0, 0] += 1
    path = tmp_path / "full_weights.pt"
    torch.save(state, path)
    with pytest.raises(ValueError):
        load_full_weights(target, path)
    for key, tensor in target.state_dict().items():
        torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)


def test_checkpoint_step_requires_production_directory_identity():
    assert (
        checkpoint_step(
            Path("/run/global_step_4/actor/model_state_dict/full_weights.pt")
        )
        == 4
    )
    with pytest.raises(ValueError, match="global_step"):
        checkpoint_step(Path("/run/full_weights.pt"))


def test_config_preserves_native_input_contract_and_rejects_padding(tmp_path):
    model = {
        "model_type": "gr00t",
        "precision": "bf16",
        "action_head_precision": "fp32",
        "inference_autocast_dtype": "bf16",
        "obs_converter_type": "libero_grouped",
        "embodiment_projector_id": 31,
        "opd_fixed_text_padding": False,
        "disable_image_tiling": True,
        "num_action_chunks": 8,
        "denoising_steps": 4,
        "model_path": "/base",
        "rl_head_config": {"noise_method": "flow_ode", "disable_dropout": True},
    }
    cfg = OmegaConf.create(
        {"actor": {"model": model}, "algorithm": {"flow_refresh_updates": 2}}
    )
    path = tmp_path / "resolved_config.yaml"
    OmegaConf.save(cfg, path)
    actual, reuse = candidate_model_config(path)
    assert actual.model_path == "/base" and reuse == 2
    cfg.actor.model.opd_fixed_text_padding = True
    OmegaConf.save(cfg, path)
    with pytest.raises(ValueError, match="opd_fixed_text_padding"):
        candidate_model_config(path)


def test_unchanged_heads_and_changed_non_backbone_version_are_reported(tmp_path):
    target = policy()
    target.register_buffer("version", torch.tensor(0))
    state = {key: value.clone() for key, value in target.state_dict().items()}
    state["version"] += 1
    path = tmp_path / "full_weights.pt"
    torch.save(state, path)
    result = load_full_weights(target, path)["base_comparison"]
    assert result["changed_action_head_tensors"] == 0
    assert result["unchanged_action_head_tensors"] == 2
    assert result["unchanged_backbone_tensors"] == 2
    assert result["other_changed_state_keys"] == ["version"]
    assert target.version.item() == 1


def test_baseline_preserves_weights_and_records_content_without_strict_reload(tmp_path):
    model = policy()
    before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "experiment_cfg").mkdir()
    (tmp_path / "experiment_cfg/metadata.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"base checkpoint")
    result = prepare_evaluation_weights(model, None, tmp_path)
    assert result["baseline"] and result["strict"] is False
    assert result["weights"] is result["weights_sha256"] is None
    assert len(result["base_checkpoint_identity"]["files"]) == 3
    assert "base_comparison" not in result
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)


def test_trained_source_still_requires_strict_load_and_identifies_weights(tmp_path):
    model = policy()
    path = tmp_path / "full_weights.pt"
    torch.save(model.state_dict(), path)
    result = prepare_evaluation_weights(model, path, tmp_path)
    assert result["strict"] and result["baseline"] is False
    assert result["weights"] == str(path.resolve())
    assert len(result["weights_sha256"]) == 64
    assert "base_checkpoint_identity" not in result


@pytest.mark.parametrize("source", [["--baseline"], ["--weights", "weights.pt"]])
def test_cli_accepts_explicit_baseline_or_existing_weights_mode(source):
    args = parse_args(["--config", "config.yaml", "--output", "new-output", *source])
    assert args.baseline == (source == ["--baseline"])


@pytest.mark.parametrize("source", [[], ["--baseline", "--weights", "weights.pt"]])
def test_cli_rejects_missing_or_ambiguous_policy_source(source):
    with pytest.raises(SystemExit):
        parse_args(["--config", "config.yaml", "--output", "new-output", *source])
