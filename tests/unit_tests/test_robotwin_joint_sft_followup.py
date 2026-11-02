# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Pure CPU checks for downstream gate, result provenance, and fresh MOPD config."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.config import validate_flow_refresh_cfg, validate_rollout_allocation_cfg
from toolkits.robotwin_mopd.joint_sft_followup import (
    TASKS,
    compare_reports,
    exact_process_alive,
    mopd_command,
    validate_report,
    viability_gate,
)


def report(counts):
    return {
        "tasks": {
            task: {"episodes": 8, "successes": n, "success_rate": n / 8}
            for task, n in zip(TASKS, counts)
        },
        "episodes": [
            {
                "task": task,
                "actual_reset_seed": task_index * 100 + i,
                "success_once": i < n,
            }
            for task_index, (task, n) in enumerate(zip(TASKS, counts))
            for i in range(8)
        ],
    }


@pytest.mark.parametrize(
    "counts,passed",
    [
        ([8, 0, 0, 0], False),
        ([1, 1, 0, 0], True),
        ([0, 0, 0, 0], False),
        ([0, 1, 0, 1], True),
    ],
)
def test_predeclared_viability_gate(counts, passed):
    assert viability_gate(report(counts))["passed"] is passed


def test_summary_cannot_hide_missing_actual_episodes():
    value = report([8, 8, 8, 8])
    value["episodes"].pop()
    with pytest.raises(ValueError, match="32 actual"):
        validate_report(value)


def test_summary_cannot_claim_unrecorded_success():
    value = report([1, 1, 0, 0])
    value["tasks"][TASKS[0]]["successes"] = 8
    with pytest.raises(ValueError, match="summary"):
        viability_gate(value)


def test_seed_comparison_does_not_assume_action_noise_matches():
    initial = report([1, 1, 0, 0])
    final = report([2, 1, 0, 0])
    result = compare_reports(initial, final)
    assert result["matched_actual_scene_seed_sequence"] is True
    assert result["action_noise_fixed"] is False
    assert result["tasks"][TASKS[0]]["success_rate_change"] == 0.125
    final = copy.deepcopy(final)
    final["episodes"][0]["actual_reset_seed"] += 10000
    result = compare_reports(initial, final)
    assert result["matched_actual_scene_seed_sequence"] is False
    assert result["comparison_kind"].startswith("Unmatched scenes")


def test_pid_reuse_does_not_count_as_upstream_alive(monkeypatch):
    monkeypatch.setattr(
        "psutil.Process",
        lambda pid: SimpleNamespace(
            create_time=lambda: 200.0, status=lambda: "running"
        ),
    )
    assert not exact_process_alive({"pid": 123, "create_time": 100.0})
    assert exact_process_alive({"pid": 123, "create_time": 200.0})


def test_mopd_initialization_is_sft500_fresh_optimizer(monkeypatch, tmp_path):
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(repo / "examples/embodiment"))
    monkeypatch.setenv("ROBOTWIN_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("ASSETS_PATH", str(tmp_path / "assets"))
    monkeypatch.setenv("OPD_RUN_DIR", str(tmp_path / "run"))
    checkpoint = tmp_path / "checkpoints/global_step_500"
    command = mopd_command(tmp_path, checkpoint)
    overrides = command[command.index("--output-root") + 2 :]
    with initialize_config_dir(
        version_base="1.1", config_dir=str(repo / "examples/embodiment/config")
    ):
        cfg = compose(config_name="robotwin4_mopd_pi05", overrides=overrides)
    OmegaConf.resolve(cfg)
    validate_flow_refresh_cfg(cfg)
    validate_rollout_allocation_cfg(cfg)
    assert cfg.runner.resume_dir is None and cfg.runner.ckpt_path is None
    assert cfg.actor.model.model_path == str(checkpoint)
    assert cfg.runner.max_steps == 80 and cfg.runner.save_interval == 20
    assert cfg.algorithm.flow_refresh_updates == 4
    assert cfg.env.train.total_num_envs == 32
    assert cfg.actor.global_batch_size == 128 and cfg.actor.micro_batch_size == 8
    assert cfg.actor.optim.lr == 1e-5 and cfg.actor.optim.lr_warmup_steps == 4
    assert cfg.actor.model.openpi.nft_full_horizon
    assert cfg.runner.logger.wandb_group == "robotwin_joint_sft_mopd_pilots"
    assert cfg.actor.cast_trainable_params_to_fp32


@pytest.mark.parametrize(
    "status,alive,raises",
    [
        ("FAILED", True, True),
        ("CONTROL_REVIEW_REQUIRED", False, True),
        ("TRAINING", False, True),
        ("COMPLETE", False, False),
    ],
)
def test_upstream_exit_requires_complete(monkeypatch, tmp_path, status, alive, raises):
    import json

    from toolkits.robotwin_mopd.joint_sft_followup import ReviewRequired, upstream_state

    path = tmp_path / "status.json"
    path.write_text(json.dumps({"pid": 123, "status": status}))
    monkeypatch.setattr(
        "toolkits.robotwin_mopd.joint_sft_followup.exact_process_alive",
        lambda identity: alive,
    )
    if raises:
        with pytest.raises(ReviewRequired):
            upstream_state({"pid": 123, "create_time": 1.0}, path)
    else:
        assert upstream_state({"pid": 123, "create_time": 1.0}, path)[1] is False


def test_small_peer_compute_process_still_blocks_gpu_start(monkeypatch):
    from toolkits.robotwin_mopd.joint_sft_followup import BusyGPUs, require_idle

    monkeypatch.setattr(
        "toolkits.robocasa_mopd.owned_ray.check_gpus", lambda count, env: None
    )

    def query(command, **kwargs):
        if "--query-gpu=uuid" in command:
            return "GPU-allowed\n"
        return "GPU-allowed, 98765\n"

    monkeypatch.setattr("subprocess.check_output", query)
    with pytest.raises(BusyGPUs, match="compute processes"):
        require_idle(1, {"CUDA_VISIBLE_DEVICES": "4"})


def test_duplicate_actual_scene_seeds_are_counted_without_changing_gate():
    value = report([1, 1, 0, 0])
    value["episodes"][0]["actual_reset_seed"] = 1
    value["episodes"][4]["actual_reset_seed"] = 5
    gate = viability_gate(value)
    assert gate["passed"] is True
    assert gate["episode_counts"][TASKS[0]] == 8
    assert gate["unique_actual_scene_seeds_per_task"][TASKS[0]] == 6
    assert gate["duplicate_actual_scene_seeds_allowed"] is True
    assert "not necessarily eight independent scenes" in gate["sampling_note"]
    result = compare_reports(value, value)
    assert result["matched_actual_scene_seed_sequence"] is True
    for side in ("initial_scene_provenance", "final_scene_provenance"):
        assert result[side]["episode_counts"][TASKS[0]] == 8
        assert result[side]["unique_actual_scene_seeds_per_task"][TASKS[0]] == 6
        assert all(
            result[side]["unique_actual_scene_seeds_per_task"][task] == 8
            for task in TASKS[1:]
        )


def test_recovery_queue_paths_preserve_original_identity(tmp_path):
    from toolkits.robotwin_mopd.joint_sft_followup import parse_args, queue_files

    original = parse_args(["--base", str(tmp_path)])
    assert original.queue_tag == "20261101"
    assert original.gpu_drain_grace_seconds == 0
    before = queue_files(tmp_path, original.queue_tag)
    for path in before:
        path.parent.mkdir(exist_ok=True)
        path.write_text("original failed queue evidence")
    recovered = parse_args(
        [
            "--base",
            str(tmp_path),
            "--queue-tag",
            "recovery_20261101",
            "--upstream-launch",
            "/new/launch.json",
            "--upstream-status",
            "/new/status.json",
            "--gpu-drain-grace-seconds",
            "120",
        ]
    )
    after = queue_files(tmp_path, recovered.queue_tag)
    assert set(before).isdisjoint(after)
    assert recovered.upstream_launch == Path("/new/launch.json")
    assert recovered.upstream_status == Path("/new/status.json")
    assert all(path.read_text() == "original failed queue evidence" for path in before)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--queue-tag", "../old"],
        ["--upstream-launch", "/new/launch.json"],
        ["--gpu-drain-grace-seconds", "601"],
    ],
)
def test_recovery_options_fail_before_queue_state_creation(tmp_path, arguments):
    from toolkits.robotwin_mopd.joint_sft_followup import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--base", str(tmp_path), *arguments])
    assert not list(tmp_path.iterdir())


def test_gpu_grace_is_explicit_bounded_and_preserves_strict_check(monkeypatch):
    from toolkits.robotwin_mopd import joint_sft_followup as module

    clock = [0.0]
    checks = []
    waits = []
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        module.time, "sleep", lambda duration: clock.__setitem__(0, clock[0] + duration)
    )

    def busy_then_idle(count, env):
        checks.append((count, env))
        if clock[0] < 20:
            raise module.BusyGPUs("still occupied")

    monkeypatch.setattr(module, "require_idle", busy_then_idle)
    with pytest.raises(module.BusyGPUs):
        module.wait_for_idle(4, {}, 0, 100)
    assert clock[0] == 0 and len(checks) == 1
    module.wait_for_idle(
        4, {}, 30, 100, on_wait=lambda reason, remaining: waits.append(remaining)
    )
    assert clock[0] == 20 and waits == [30, 20]
    assert len(checks) == 4


def test_busy_gpu_grace_expiry_does_not_relax_or_reassign(monkeypatch):
    from toolkits.robotwin_mopd import joint_sft_followup as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        module.time, "sleep", lambda duration: clock.__setitem__(0, clock[0] + duration)
    )

    def busy(count, env):
        assert env["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
        raise module.BusyGPUs("peer still present")

    monkeypatch.setattr(module, "require_idle", busy)
    with pytest.raises(module.BusyGPUs, match="peer still present"):
        module.wait_for_idle(4, {"CUDA_VISIBLE_DEVICES": "4,5,6,7"}, 120, 15)
    assert clock[0] == 15
