# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Explicit changed-policy profile guards and historical native compatibility."""

import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from test_libero_evaluation_launcher import arguments
from test_libero_production_launcher import fixture_paths

from toolkits.libero_gr00t import launch_evaluation, launch_production
from toolkits.libero_gr00t.evaluate_full_checkpoint import candidate_model_config
from toolkits.libero_gr00t.padding_profile import student_input_contract


def production(tmp_path):
    python, models = fixture_paths(tmp_path)
    args = SimpleNamespace(
        python=python,
        output=tmp_path / "out",
        gpus=[0, 1, 2, 3],
        updates=4,
        rollout_steps=520,
        config_name="libero_mopd_candidate_m1",
    )
    paths = {
        "student": models / "student",
        **{
            f"libero_{s}": models / n
            for s, n in [
                ("spatial", "spatial"),
                ("object", "object"),
                ("goal", "goal"),
                ("10", "long"),
            ]
        },
    }
    return args, paths


def test_native_profile_is_config_noop_and_historical_reload_works(tmp_path):
    args, paths = production(tmp_path)
    original = launch_production.build_config(args, paths)
    args.student_padding_profile = "native"
    explicit = launch_production.build_config(args, paths)
    assert OmegaConf.to_container(original) == OmegaConf.to_container(explicit)
    assert explicit.actor.model.opd_canonical_fixed_padding is False
    assert (
        explicit.actor.global_batch_size == 64 and explicit.actor.micro_batch_size == 1
    )
    cfg_path = tmp_path / "native.yaml"
    del explicit.actor.model.opd_canonical_fixed_padding
    OmegaConf.save(explicit, cfg_path)
    model, _ = candidate_model_config(cfg_path)
    assert student_input_contract(model)["opd_canonical_fixed_padding"] is False


def test_canonical_profile_is_explicit_and_teachers_keep_legacy_fixed_policy(tmp_path):
    args, paths = production(tmp_path)
    args.student_padding_profile = "canonical-fixed570"
    cfg = launch_production.build_config(args, paths)
    assert cfg.actor.global_batch_size == 256 and cfg.actor.micro_batch_size == 4
    assert cfg.actor.global_batch_size // (4 * cfg.actor.micro_batch_size) == 16
    for model in [cfg.actor.model, cfg.rollout.model]:
        assert (
            student_input_contract(model, args.student_padding_profile)[
                "fixed_text_length"
            ]
            == 570
        )
    for expert in cfg.rollout.expert_models.values():
        merged = OmegaConf.merge(cfg.actor.model, expert)
        assert merged.opd_fixed_text_padding is True
        assert merged.opd_canonical_fixed_padding is False
        assert merged.action_head_precision == "bf16"
    cfg_path = tmp_path / "canonical.yaml"
    OmegaConf.save(cfg, cfg_path)
    with pytest.raises(ValueError, match="opd_fixed_text_padding"):
        candidate_model_config(cfg_path)
    model, _ = candidate_model_config(
        cfg_path, student_padding_profile=args.student_padding_profile
    )
    assert model.opd_canonical_fixed_padding
    cfg.rollout.model.opd_canonical_fixed_padding = False
    OmegaConf.save(cfg, cfg_path)
    with pytest.raises(ValueError, match="opd_canonical_fixed_padding"):
        candidate_model_config(
            cfg_path, student_padding_profile=args.student_padding_profile
        )


@pytest.mark.parametrize("arm", [0, 2, 3])
def test_canonical_training_profile_rejects_other_arms(tmp_path, arm):
    args, paths = production(tmp_path)
    args.config_name = f"libero_mopd_candidate_m{arm}"
    args.student_padding_profile = "canonical-fixed570"
    with pytest.raises(ValueError, match="M1/R2"):
        launch_production.build_config(args, paths)


def test_distributed_eval_requires_explicit_matching_canonical_identity(tmp_path):
    args = arguments(tmp_path)
    native = launch_evaluation.build_config(args)
    args.student_padding_profile = "canonical-fixed570"
    cfg = launch_evaluation.build_config(args)
    identity = student_input_contract(cfg.rollout.model, args.student_padding_profile)
    assert (
        launch_evaluation.validate_checkpoint(args, cfg)["student_input_contract"]
        == identity
    )
    args.weights = tmp_path / "weights.pt"
    args.weights.write_bytes(b"weights")
    args.validation_protocol = tmp_path / "protocol.json"
    protocol = {
        "strict": True,
        "state_tensor_count": 899,
        "base_comparison": {"unchanged_backbone_tensors": 585},
        "weights_sha256": launch_evaluation.sha256(args.weights),
        "metadata_sha256": launch_evaluation.sha256(
            args.student / "experiment_cfg/metadata.json"
        ),
        "base_checkpoint": str(args.student),
        "model_config": OmegaConf.to_container(native.rollout.model, resolve=True),
    }
    args.validation_protocol.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="opd_fixed_text_padding"):
        launch_evaluation.validate_checkpoint(args, cfg)
    protocol["model_config"] = OmegaConf.to_container(cfg.rollout.model, resolve=True)
    args.validation_protocol.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="padding profile"):
        launch_evaluation.validate_checkpoint(args, cfg)
    protocol.update(
        student_padding_profile=args.student_padding_profile,
        student_input_contract=identity,
    )
    args.validation_protocol.write_text(json.dumps(protocol))
    assert not launch_evaluation.validate_checkpoint(args, cfg)["baseline"]
