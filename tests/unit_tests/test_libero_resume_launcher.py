# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Validate explicit quota-boundary continuation without launching GPU workers."""

import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from toolkits.libero_gr00t import launch_production as launch


def resume_fixture(tmp_path, monkeypatch):
    run = tmp_path / "source_run"
    run.mkdir()
    args = SimpleNamespace(
        python=tmp_path / "runtime/bin/python",
        output=run,
        gpus=[0, 1, 6, 7],
        updates=40,
        save_every_updates=20,
        rollout_steps=520,
        config_name="libero_mopd_candidate_m1",
        allocation_smoke=False,
        resume_checkpoint=None,
    )
    paths = {
        name: tmp_path / name
        for name in (
            "student",
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
        )
    }
    source = launch.build_config(args, paths)
    OmegaConf.save(source, run / "resolved_config.yaml")
    checkpoint = run / "libero_production_smoke/checkpoints/global_step_20"
    checkpoint.mkdir(parents=True)
    state = {
        "step_unit": "optimizer_update",
        "optimizer_updates": 20,
        "rollout_count": 10,
        "env_steps": 94540,
    }
    allocation = {
        "rounds": 10,
        "trajectories": 320,
        "simulation_steps": 94540,
        "config": OmegaConf.to_container(
            source.runner.rollout_allocation, resolve=True
        ),
    }
    (checkpoint / "runner_state.json").write_text(json.dumps(state))
    (checkpoint / "rollout_allocation_state.json").write_text(json.dumps(allocation))
    monkeypatch.setattr(
        launch, "inspect_resume_dcp", lambda path, step: {"scheduler_last_epoch": step}
    )
    args.output = tmp_path / "continued_run"
    args.gpus = [2, 3, 4, 5]
    args.updates = 80
    args.resume_checkpoint = checkpoint
    return args, paths, source, state, allocation


def test_resume_preserves_recipe_and_continues_total_update_budget(
    tmp_path, monkeypatch
):
    args, paths, _, _, _ = resume_fixture(tmp_path, monkeypatch)
    cfg = launch.build_config(args, paths)
    assert cfg.runner.resume_mode == "adaptive_round_boundary"
    assert cfg.runner.resume_dir == str(args.resume_checkpoint)
    assert cfg.runner.max_steps == 80
    assert cfg.cluster.component_placement["actor,env,rollout"].placement == "2,3,4,5"
    assert args.resume_summary["starting_optimizer_update"] == 20
    assert args.resume_summary["additional_optimizer_updates"] == 60
    assert args.resume_summary["additional_rollout_rounds"] == 30
    assert args.resume_summary["exact_replay"] is False


def test_resume_requires_same_actor_cache_settings(tmp_path, monkeypatch):
    args, paths, source, _, _ = resume_fixture(tmp_path, monkeypatch)
    args.actor_cache_retention_gib = 28
    args.actor_cache_preallocate_gib = 26
    with pytest.raises(ValueError, match="recipe changed at actor"):
        launch.build_config(args, paths)
    OmegaConf.update(source, "actor.cuda_cache_retention_gib", 28.0, force_add=True)
    OmegaConf.update(source, "actor.cuda_cache_preallocate_gib", 26.0, force_add=True)
    OmegaConf.save(source, args.resume_checkpoint.parents[2] / "resolved_config.yaml")
    cfg = launch.build_config(args, paths)
    assert cfg.actor.cuda_cache_preallocate_gib == 26
    args.actor_cache_preallocate_gib = 24
    with pytest.raises(ValueError, match="recipe changed at actor"):
        launch.build_config(args, paths)


def test_resume_requires_same_microbatch_sync_mode(tmp_path, monkeypatch):
    args, paths, source, _, _ = resume_fixture(tmp_path, monkeypatch)
    args.sync_every_microbatch = True
    with pytest.raises(ValueError, match="recipe changed at actor"):
        launch.build_config(args, paths)
    source.actor.fsdp_config.enable_gradient_accumulation = False
    OmegaConf.save(source, args.resume_checkpoint.parents[2] / "resolved_config.yaml")
    cfg = launch.build_config(args, paths)
    assert cfg.actor.fsdp_config.enable_gradient_accumulation is False
    args.sync_every_microbatch = False
    with pytest.raises(ValueError, match="recipe changed at actor"):
        launch.build_config(args, paths)


@pytest.mark.parametrize(
    "key,value",
    [
        ("actor.global_batch_size", 128),
        ("actor.micro_batch_size", 2),
        ("actor.optim.lr", 1e-5),
        ("actor.optim.lr_warmup_steps", 8),
        ("actor.model.action_head_precision", "bf16"),
        ("actor.model.model_path", "/other"),
        ("algorithm.kl_beta", 0.5),
        ("algorithm.flow_refresh_updates", 4),
        ("rollout.expert_models.libero_spatial.model_path", "/other_teacher"),
        ("actor.fsdp_config.amp_autocast.enabled", False),
        ("env.train.total_num_envs", 48),
        ("env.train.seed", 1234),
        ("runner.rollout_allocation.rho", 0.75),
    ],
)
def test_changed_learning_recipe_rejected(tmp_path, monkeypatch, key, value):
    args, paths, source, _, _ = resume_fixture(tmp_path, monkeypatch)
    OmegaConf.update(source, key, value, force_add=True)
    OmegaConf.save(source, args.resume_checkpoint.parents[2] / "resolved_config.yaml")
    with pytest.raises(ValueError, match="recipe changed"):
        launch.build_config(args, paths)


@pytest.mark.parametrize(
    "fault",
    [
        "target",
        "epoch_cap",
        "step",
        "round",
        "simulation",
        "trajectories",
        "controller",
    ],
)
def test_inconsistent_or_nonadvancing_resume_rejected(tmp_path, monkeypatch, fault):
    args, paths, _, state, allocation = resume_fixture(tmp_path, monkeypatch)
    if fault == "target":
        args.updates = 20
    elif fault == "epoch_cap":
        args.updates = 2000
    elif fault == "step":
        state["optimizer_updates"] = 19
    elif fault == "round":
        allocation["rounds"] = 9
    elif fault == "simulation":
        allocation["simulation_steps"] -= 1
    elif fault == "trajectories":
        allocation["trajectories"] = 300
    else:
        allocation["config"]["rho"] = 0.75
    (args.resume_checkpoint / "runner_state.json").write_text(json.dumps(state))
    (args.resume_checkpoint / "rollout_allocation_state.json").write_text(
        json.dumps(allocation)
    )
    with pytest.raises(ValueError):
        launch.build_config(args, paths)


def test_allocation_smoke_cannot_be_resumed(tmp_path, monkeypatch):
    args, paths, _, _, _ = resume_fixture(tmp_path, monkeypatch)
    args.allocation_smoke = True
    with pytest.raises(ValueError, match="cannot be combined"):
        launch.build_config(args, paths)


def test_real_small_dcp_checks_scheduler_and_detects_truncated_shard(tmp_path):
    from torch.distributed import checkpoint as dcp

    state = {
        "fsdp_checkpoint": {
            "model": {"weight": torch.ones(2)},
            "optimizers": {"state": {"weight": {"exp_avg": torch.zeros(2)}}},
            "lr_schedulers": [{"last_epoch": 20}],
        }
    }
    path = tmp_path / "dcp"
    dcp.save(state, checkpoint_id=path)
    result = launch.inspect_resume_dcp(path, 20)
    assert result["scheduler_last_epoch"] == 20
    assert len(result["metadata_sha256"]) == 64
    with pytest.raises(ValueError, match="scheduler epoch"):
        launch.inspect_resume_dcp(path, 18)
    shard = next(path.glob("*.distcp"))
    shard.write_bytes(shard.read_bytes()[:100])
    with pytest.raises(ValueError, match="truncated"):
        launch.inspect_resume_dcp(path, 20)


def test_native_resume_accepts_only_missing_or_boolean_false_canonical_flag(
    tmp_path, monkeypatch
):
    args, paths, source, _, _ = resume_fixture(tmp_path, monkeypatch)
    del source.actor.model.opd_canonical_fixed_padding
    config_path = args.resume_checkpoint.parents[2] / "resolved_config.yaml"
    OmegaConf.save(source, config_path)
    cfg = launch.build_config(args, paths)
    assert cfg.actor.model.opd_canonical_fixed_padding is False
    for invalid in [True, 0, "false"]:
        OmegaConf.update(
            source, "actor.model.opd_canonical_fixed_padding", invalid, force_add=True
        )
        OmegaConf.save(source, config_path)
        with pytest.raises(ValueError, match="recipe changed at actor"):
            launch.build_config(args, paths)


def test_resume_cannot_change_any_seed_with_explicit_override(tmp_path, monkeypatch):
    args, paths, _, _, _ = resume_fixture(tmp_path, monkeypatch)
    args.seed = 0
    cfg = launch.build_config(args, paths)
    assert (
        cfg.actor.seed
        == cfg.rollout.seed
        == cfg.env.train.seed
        == cfg.runner.rollout_allocation.seed
        == 0
    )
    args.seed = 1
    with pytest.raises(ValueError, match="Resume recipe changed at actor"):
        launch.build_config(args, paths)


def test_learning_rate_override_cannot_change_resume_optimizer(tmp_path, monkeypatch):
    args, paths, source, _, _ = resume_fixture(tmp_path, monkeypatch)
    args.learning_rate = float(source.actor.optim.lr)
    cfg = launch.build_config(args, paths)
    assert cfg.actor.optim.lr == cfg.actor.optim.value_lr == args.learning_rate
    args.learning_rate = 3e-5
    with pytest.raises(ValueError, match="Resume recipe changed at actor"):
        launch.build_config(args, paths)


def test_global_batch_override_requires_original_resume_batch(tmp_path, monkeypatch):
    args, paths, source, _, _ = resume_fixture(tmp_path, monkeypatch)
    args.global_batch_size = int(source.actor.global_batch_size)
    cfg = launch.build_config(args, paths)
    assert cfg.actor.global_batch_size == args.global_batch_size
    args.global_batch_size = 256
    with pytest.raises(ValueError, match="Resume recipe changed at actor"):
        launch.build_config(args, paths)
    source.actor.global_batch_size = 256
    OmegaConf.save(source, args.resume_checkpoint.parents[2] / "resolved_config.yaml")
    cfg = launch.build_config(args, paths)
    assert cfg.actor.global_batch_size == 256
    assert cfg.actor.micro_batch_size == 1
    args.global_batch_size = None
    with pytest.raises(ValueError, match="Resume recipe changed at actor"):
        launch.build_config(args, paths)
