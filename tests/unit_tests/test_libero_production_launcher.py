# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Launch planning and ownership checks without running Ray or using GPUs."""

import json
import sys
from types import SimpleNamespace

import pytest

from toolkits.libero_gr00t import launch_production as launcher


def fixture_paths(tmp_path):
    python = tmp_path / "runtime/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("unused test wrapper")
    (python.parent.parent / "env.sh").write_text("# runtime fixture")
    models = tmp_path / "models"
    for name in ["student", "spatial", "object", "goal", "long"]:
        path = models / name
        (path / "experiment_cfg").mkdir(parents=True)
        (path / "config.json").write_text("{}")
        (path / "experiment_cfg/metadata.json").write_text("{}")
    return python, models


def test_config_uses_physical_ids_and_labels_incomplete_rollout(tmp_path):
    python, models = fixture_paths(tmp_path)
    args = SimpleNamespace(
        python=python,
        output=tmp_path / "out",
        gpus=[4, 0, 1, 3],
        updates=2,
        rollout_steps=32,
    )
    paths = {
        "student": models / "student",
        "libero_spatial": models / "spatial",
        "libero_object": models / "object",
        "libero_goal": models / "goal",
        "libero_10": models / "long",
    }
    cfg = launcher.build_config(args, paths)
    assert cfg.cluster.component_placement["actor,env,rollout"].placement == "0,1,3,4"
    assert cfg.cluster.node_groups[0].env_configs[0].python_interpreter_path == str(
        python
    )
    assert cfg.runner.max_steps == cfg.runner.save_interval == 2
    assert set(cfg.env.train.task_suite_horizons.values()) == {32}
    assert cfg.env.train.total_num_envs == 16
    args.rollout_steps = 520
    cfg = launcher.build_config(args, paths)
    assert list(cfg.env.train.task_suite_horizons.values()) == [240, 320, 320, 520]


@pytest.mark.parametrize(
    "run,fail_driver", [(False, False), (True, False), (True, True)]
)
def test_default_dry_run_and_explicit_run_are_owner_scoped(
    tmp_path, monkeypatch, run, fail_driver
):
    python, models = fixture_paths(tmp_path)
    output = tmp_path / "out"
    argv = [
        "launch_production",
        "--python",
        str(python),
        "--models",
        str(models),
        "--student",
        str(models / "student"),
        "--spatial-teacher",
        str(models / "spatial"),
        "--output",
        str(output),
        "--gpus",
        "4",
        "0",
        "1",
        "3",
    ] + (["--run"] if run else [])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("RAY_ADDRESS", "other-cluster:6379")
    monkeypatch.setattr(
        launcher,
        "read_runtime_environment",
        lambda _: {"PYTHONPATH": str(launcher.REPO), "LIBERO_CONFIG_PATH": "/fixture"},
    )
    ray_dir = tmp_path / "private-ray"
    ray_dir.mkdir()
    def short_ray_directory(**kwargs):
        # Ray appends session and socket names; Linux AF_UNIX permits 107 bytes.
        example = "/tmp/" + kwargs["prefix"] + "abcdefgh/ray/session_2026-11-04_23-35-31_073485_2878406/sockets/plasma_store"
        assert len(example.encode()) <= 107
        return str(ray_dir)

    monkeypatch.setattr(launcher.tempfile, "mkdtemp", short_ray_directory)
    ports = {name: 19000 + i for i, name in enumerate(launcher.PORT_NAMES)}
    monkeypatch.setattr(launcher, "choose_ports", lambda: ports)
    calls, cleanups, checks = [], [], []
    monkeypatch.setattr(launcher.signal, "signal", lambda *args: None)

    def check_gpus(count, env):
        checks.append((count, env["GPU_FREE_MB"]))
        return "GPU preflight fixture\n"

    def run_logged(command, env, log):
        calls.append((command, env.copy()))
        log.write_text("fixture command\n")
        if "--head" in command:
            assert "RAY_ADDRESS" not in env
            assert launcher.OWNER_KEY in env
            (ray_dir / "ray/ray_current_cluster").write_text("127.0.0.1:19000")
        else:
            assert env["RAY_ADDRESS"] == "127.0.0.1:19000"
            assert launcher.OWNER_KEY not in env
            if fail_driver:
                raise RuntimeError("simulated driver failure")

    monkeypatch.setattr(launcher, "check_gpus", check_gpus)
    monkeypatch.setattr(launcher, "run_logged", run_logged)
    monkeypatch.setattr(launcher, "stop_owned", cleanups.append)
    if fail_driver:
        with pytest.raises(RuntimeError, match="simulated"):
            launcher.main()
    else:
        launcher.main()
    manifest = json.loads((output / "launch.json").read_text())
    assert manifest["physical_gpus"] == [0, 1, 3, 4]
    assert manifest["incomplete_episodes"] and not manifest["performance_evidence"]
    assert (output / "resolved_config.yaml").is_file()
    if run:
        assert len(calls) == 2
        assert checks == [(4, "30720")]
        assert cleanups == [manifest["owner"]]
        expected = "FAILED" if fail_driver else "OK"
        assert (output / "status").read_text().strip() == expected
        assert all("stop" not in command for command, _ in calls)
    else:
        assert calls == cleanups == checks == []
        assert (output / "status").read_text().strip() == "DRY_RUN"


def test_duplicate_gpu_ids_rejected_before_launch(tmp_path):
    args = SimpleNamespace(gpus=[0, 0, 1, 2], updates=2, rollout_steps=32)
    with pytest.raises(ValueError, match="distinct"):
        launcher.build_config(args, {})


def test_real_placement_keeps_physical_ids_on_an_eight_gpu_node(tmp_path):
    from omegaconf import OmegaConf

    from rlinf.scheduler.cluster.node import NodeGroupInfo, NodeInfo
    from rlinf.scheduler.hardware import Accelerator, HardwareInfo, HardwareResource
    from rlinf.utils.placement import HybridComponentPlacement

    node = NodeInfo(
        node_labels=[],
        node_rank=0,
        ray_id="fake",
        node_ip="127.0.0.1",
        num_cpus=16,
        python_interpreter_path="/fixture/python",
        default_env_vars={},
        env_vars={},
        hardware_resources=[
            HardwareResource(
                type=Accelerator.HW_TYPE,
                infos=[HardwareInfo(type=Accelerator.HW_TYPE, model="NV_GPU:Mock")] * 8,
            )
        ],
    )
    group = NodeGroupInfo(
        label="libero_smoke", nodes=[node], hardware_type=Accelerator.HW_TYPE
    )
    # Hardware discovery uses all eight NVML devices, independently of Ray's
    # four logical GPU resource units. No Ray instance is needed for placement.
    cluster = SimpleNamespace(
        get_node_group=lambda label=None: group, get_node_info=lambda rank: node
    )
    cfg = OmegaConf.create(
        {
            "cluster": {
                "component_placement": {
                    "actor,env,rollout": {
                        "node_group": "libero_smoke",
                        "placement": "0,1,4,5",
                    }
                }
            }
        }
    )
    placement = HybridComponentPlacement(cfg, cluster)
    assert placement.get_world_size("actor") == 4
    resolved = placement.get_strategy("actor").get_placement(cluster)
    assert [p.visible_accelerators for p in resolved] == [["0"], ["1"], ["4"], ["5"]]


def test_reused_cleanup_selector_excludes_peer_processes(monkeypatch):
    import os

    from toolkits.robocasa_mopd import owned_ray

    def process(pid, uid, owner):
        return SimpleNamespace(
            pid=pid,
            uids=lambda: SimpleNamespace(real=uid),
            status=lambda: "running",
            environ=lambda: {owned_ray.OWNER_KEY: owner},
        )

    mine = process(os.getpid() + 100, os.getuid(), "this-launch")
    peers = [
        process(os.getpid() + 101, os.getuid(), "another-launch"),
        process(os.getpid() + 102, os.getuid() + 1, "this-launch"),
        process(os.getpid(), os.getuid(), "this-launch"),
    ]
    monkeypatch.setattr(owned_ray.psutil, "process_iter", lambda: [mine, *peers])
    assert owned_ray.owned_processes("this-launch") == [mine]
