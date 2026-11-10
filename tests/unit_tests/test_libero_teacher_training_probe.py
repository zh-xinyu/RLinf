# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU guards for a no-actor probe of the actual production train collector."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rlinf.envs.libero.multi_suite import validate_rollout_plan
from rlinf.scheduler.cluster.config import ClusterConfig
from toolkits.libero_gr00t import teacher_training_probe as probe
from toolkits.libero_gr00t.launch_production import build_config


@pytest.fixture
def source(tmp_path):
    args = SimpleNamespace(
        python=tmp_path / "python",
        output=tmp_path / "source",
        gpus=[0, 1, 2, 3],
        updates=2,
        save_every_updates=2,
        rollout_steps=520,
        config_name="libero_mopd_candidate_m1",
    )
    paths = {name: tmp_path / name for name in ("student", *probe.SUITES)}
    return build_config(args, paths)


def reference(tasks=(0, 1), trials=(0, 1)):
    return {
        "seed": 1234,
        "episodes": [
            {
                "suite": suite,
                "task_id": task,
                "trial": trial,
                "success_once": False,
                "steps": probe.HORIZONS[suite],
            }
            for suite in probe.SUITES
            for task in tasks
            for trial in trials
        ],
    }


def layouts():
    return [
        {
            "worker": rank,
            "envs": 1,
            "epochs": 1,
            "tasks": [
                {
                    "task_id": task,
                    "task": f"{suite}/{task}",
                    "suite": suite,
                    "horizon": probe.HORIZONS[suite],
                }
                for task in range(10)
            ],
        }
        for rank, suite in enumerate(probe.SUITES)
    ]


def records():
    result = []
    for trial in (0, 1):
        for task in (0, 1):
            for row in probe.make_plan(
                layouts(), task, trial, 1 + 2 * trial + task, 1234
            ):
                result.append(
                    {
                        **row,
                        "trial_id": trial,
                        "success_once": False,
                        "simulation_steps": row["horizon"],
                    }
                )
    return result


def test_rank_native_contract_and_real_train_plan_validation(source, tmp_path):
    before = OmegaConf.to_container(source, resolve=True)
    cfg = probe.configure_probe(
        source,
        tmp_path / "out",
        tmp_path / "python",
        [4, 5, 6, 7],
        reference(),
        [0, 1],
        [0, 1],
        10,
        [1, 4],
        {"OPD_LIBERO_AUTOMAP_EGL": "1", "NCCL_P2P_DISABLE": "1"},
    )
    assert OmegaConf.to_container(source, resolve=True) == before
    validated = ClusterConfig.from_dict_cfg(cfg.cluster)
    assert validated.node_groups[0].env_configs[0].env_vars == [
        {"OPD_LIBERO_AUTOMAP_EGL": "1"},
        {"NCCL_P2P_DISABLE": "1"},
    ]
    assert cfg.runner.only_eval is False
    assert cfg.env.train.total_num_envs == 4
    assert cfg.env.train.task_sampling_strategy == "suite_quota"
    assert cfg.env.train.freeze_finished_episodes is True
    assert cfg.env.train.reset_settle_steps == 10
    assert cfg.runner.ckpt_path is None
    plan = probe.make_plan(layouts(), 7, 1, 2, 1234)
    for rank, suite in enumerate(probe.SUITES):
        local = probe.rank_config(cfg, rank)
        assert list(local.env.train.task_suite_names) == [suite]
        assert local.env.train.task_suite_horizons[suite] == probe.HORIZONS[suite]
        assert (
            local.actor.model.model_path
            == source.rollout.expert_models[suite].model_path
        )
        assert local.actor.model.obs_converter_type == "libero"
        assert local.actor.model.embodiment_tag == "libero_franka"
        assert local.actor.model.embodiment_projector_id == 31
        assert local.actor.model.opd_fixed_text_padding is True
        assert local.actor.model.action_head_precision == "bf16"
        assert local.actor.model.num_action_chunks == 8
        assert local.actor.model.denoising_steps == 4
        actual = validate_rollout_plan(
            [plan[rank]], layouts()[rank]["tasks"], 1, [suite]
        )
        assert actual[0]["reset_seed"] == 1234
        assert actual[0]["policy_seed"] == 1935
        assert actual[0]["occurrence"] == 1


@pytest.mark.parametrize(
    "fault", ["duplicate", "missing", "short_failure", "seed", "horizon", "unexpected"]
)
def test_incomplete_or_unpaired_episodes_fail_closed(fault):
    rows = records()
    if fault == "duplicate":
        rows.append(dict(rows[0]))
    elif fault == "missing":
        rows.pop()
    elif fault == "short_failure":
        rows[0]["simulation_steps"] -= 1
    elif fault == "seed":
        rows[0]["policy_seed"] += 1
    elif fault == "horizon":
        rows[0]["horizon"] = 520
    else:
        rows[0]["trial_id"] = 3
    with pytest.raises(ValueError):
        probe.audit_episodes(rows, reference(), [0, 1], [0, 1])


def test_coverage_report_uses_actual_training_outcomes():
    rows = records()
    rows[0].update(success_once=True, simulation_steps=37)
    report = probe.audit_episodes(rows, reference(), [0, 1], [0, 1])
    assert report["complete"] and report["episodes"] == 16
    assert report["training_successes"] == report["outcome_disagreements"] == 1
    assert report["reference_successes"] == report["optimizer_updates"] == 0


def test_reference_indices_and_seeds_are_read_from_protocol(
    source, tmp_path, monkeypatch
):
    manifest = {
        "seed": 42,
        "teacher_checkpoints": {
            suite: source.rollout.expert_models[suite].model_path
            for suite in probe.SUITES
        },
        "calibration_protocol": {
            "noise_seed": "seed + local_task_id * 100 + trial",
            "environment_seed": 42,
            "initial_state_indices": [3, 5],
        },
    }
    monkeypatch.setattr(probe, "verify_capture", lambda _: manifest)
    monkeypatch.setattr(
        probe,
        "read_suite",
        lambda folder, capture, suite: [
            {"suite": suite, "task_id": 0, "trial": trial} for trial in (3, 5)
        ],
    )
    selected = probe.select_reference(tmp_path, source, [0], [3, 5])
    assert selected["seed"] == 42 and len(selected["episodes"]) == 8
    with pytest.raises(ValueError, match="requested reset/noise"):
        probe.select_reference(tmp_path, source, [0], [0, 1])
    manifest["calibration_protocol"]["environment_seed"] = 0
    with pytest.raises(ValueError):
        probe.select_reference(tmp_path, source, [0], [3, 5])


@pytest.mark.parametrize("memory", [0, 7, 11, float("nan")])
def test_allocator_cap_guard(source, tmp_path, memory):
    with pytest.raises(ValueError, match="memory-gib"):
        probe.configure_probe(
            source,
            tmp_path,
            tmp_path / "python",
            [0, 1, 2, 3],
            reference(),
            [0],
            [0],
            memory,
            [],
            {},
        )


def test_real_training_methods_are_inherited_without_actor_path():
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
    from toolkits.libero_gr00t.teacher_training_workers import (
        TeacherProbeEnvWorker,
        TeacherProbeRolloutWorker,
    )

    assert TeacherProbeEnvWorker.interact is EnvWorker.interact
    assert TeacherProbeEnvWorker._run_interact_once is EnvWorker._run_interact_once
    assert (
        TeacherProbeEnvWorker._bootstrap_and_send_train
        is EnvWorker._bootstrap_and_send_train
    )
    assert TeacherProbeRolloutWorker.generate is MultiStepRolloutWorker.generate
    assert (
        TeacherProbeRolloutWorker.generate_one_epoch
        is MultiStepRolloutWorker.generate_one_epoch
    )


@pytest.mark.parametrize(
    "run,fail_driver", [(False, False), (True, False), (True, True)]
)
def test_private_launcher_requires_explicit_run_and_cleans_only_owner(
    source, tmp_path, monkeypatch, run, fail_driver
):
    import json
    import sys

    from toolkits.libero_gr00t import launch_teacher_training_probe as launcher

    config = tmp_path / "source.yaml"
    OmegaConf.save(source, config)
    capture = tmp_path / "reference"
    capture.mkdir()
    (capture / "manifest.json").write_text("{}")
    python = tmp_path / "python"
    python.touch()
    output = tmp_path / "out"
    argv = [
        "launch",
        "--config",
        str(config),
        "--reference",
        str(capture),
        "--python",
        str(python),
        "--output",
        str(output),
        "--gpus",
        "0",
        "1",
        "2",
        "3",
        "--tasks",
        "0",
        "--trials",
        "0",
    ]
    if run:
        argv.append("--run")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        launcher, "select_reference", lambda *args: reference((0,), (0,))
    )
    monkeypatch.setattr(
        launcher,
        "read_runtime_environment",
        lambda _: {
            "PYTHONPATH": str(launcher.REPO),
            "LIBERO_CONFIG_PATH": "/fixture",
            "OPD_LIBERO_AUTOMAP_EGL": "1",
        },
    )
    monkeypatch.setattr(launcher, "choose_ports", lambda: {"port": 19091})
    monkeypatch.setattr(launcher.signal, "signal", lambda *args: None)
    preflight, calls, stopped = [], [], []
    monkeypatch.setattr(
        launcher,
        "check_gpus",
        lambda count, env: preflight.append((count, env)) or "fixture",
    )
    monkeypatch.setattr(launcher, "stop_owned", stopped.append)

    def run_logged(command, env, log, **kwargs):
        calls.append((command, env))
        if "--head" in command:
            from pathlib import Path

            assert launcher.OWNER_KEY in env
            assert "RAY_ADDRESS" not in env
            (Path(env["RAY_TMPDIR"]) / "ray/ray_current_cluster").write_text(
                "127.0.0.1:19091"
            )
        else:
            assert launcher.OWNER_KEY not in env
            assert "teacher_training_probe" in " ".join(command)
            if fail_driver:
                raise RuntimeError("fixture driver failure")
            (output / "summary.json").write_text(
                json.dumps({"complete": True, "episodes": 4})
            )

    monkeypatch.setattr(launcher, "run_logged", run_logged)
    if fail_driver:
        with pytest.raises(RuntimeError, match="fixture driver"):
            launcher.main()
    else:
        launcher.main()
    manifest = json.loads((output / "launch.json").read_text())
    assert manifest["actor_launched"] is False
    assert manifest["expected_episodes"] == 4
    assert manifest["memory_gib"] == 10 and manifest["minimum_free_gib"] == 12
    if run:
        assert len(preflight) == 1 and len(calls) == 2
        assert stopped == [manifest["owner"]]
        assert (output / "status").read_text().strip() == (
            "FAILED" if fail_driver else "OK"
        )
    else:
        assert not preflight and not calls and not stopped
        assert (output / "status").read_text().strip() == "DRY_RUN"
