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
    "run,fail_driver,allocation_smoke",
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (False, False, True),
    ],
)
@pytest.mark.parametrize(
    "learning_rate,global_batch_size", [(None, None), (3e-5, None), (3e-5, 256)]
)
@pytest.mark.parametrize("cache_sizes", [(0, 0), (28, 26)])
@pytest.mark.parametrize("sync_every_microbatch", [False, True])
def test_default_dry_run_and_explicit_run_are_owner_scoped(
    tmp_path,
    monkeypatch,
    run,
    fail_driver,
    allocation_smoke,
    learning_rate,
    global_batch_size,
    cache_sizes,
    sync_every_microbatch,
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
    if learning_rate is not None:
        argv += ["--learning-rate", str(learning_rate)]
    if global_batch_size is not None:
        argv += ["--global-batch-size", str(global_batch_size)]
    if sync_every_microbatch:
        argv += ["--sync-every-microbatch"]
    if cache_sizes[0]:
        argv += [
            "--actor-cache-retention-gib",
            str(cache_sizes[0]),
            "--actor-cache-preallocate-gib",
            str(cache_sizes[1]),
        ]
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if allocation_smoke:
        argv += [
            "--allocation-smoke",
            "--config-name",
            "libero_mopd_candidate_m2",
            "--updates",
            "8",
            "--rollout-steps",
            "520",
            "--seed",
            "2",
        ]
        monkeypatch.setenv(
            "OPD_LIBERO_TEACHER_REFERENCE", str(tmp_path / "reference.json")
        )
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("RAY_ADDRESS", "other-cluster:6379")
    runtime_flags = (
        {
            "OPD_LIBERO_AUTOMAP_EGL": "1",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "0",
        }
        if allocation_smoke
        else {}
    )
    original_build_config = launcher.build_config

    def build_with_existing_environment(args, paths):
        from omegaconf import open_dict

        cfg = original_build_config(args, paths)
        if allocation_smoke:
            with open_dict(cfg):
                cfg.cluster.node_groups[0].env_configs[0].env_vars = [
                    {"EXISTING_ENV": "preserved"}
                ]
        return cfg

    monkeypatch.setattr(launcher, "build_config", build_with_existing_environment)
    monkeypatch.setattr(
        launcher,
        "read_runtime_environment",
        lambda _: {
            "PYTHONPATH": str(launcher.REPO),
            "LIBERO_CONFIG_PATH": "/fixture",
            **runtime_flags,
        },
    )
    ray_dir = tmp_path / "private-ray"
    ray_dir.mkdir()

    def short_ray_directory(**kwargs):
        # Ray appends session and socket names; Linux AF_UNIX permits 107 bytes.
        example = (
            "/tmp/"
            + kwargs["prefix"]
            + "abcdefgh/ray/session_2026-11-04_23-35-31_073485_2878406/sockets/plasma_store"
        )
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

    def run_logged(command, env, log, **kwargs):
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
    assert manifest["incomplete_episodes"] is not allocation_smoke
    assert not manifest["performance_evidence"] and manifest["smoke_only"]
    assert manifest["seed_override"] == (2 if allocation_smoke else None)
    assert manifest["learning_rate_override"] == learning_rate
    assert manifest["global_batch_size_override"] == global_batch_size
    from omegaconf import OmegaConf

    resolved = OmegaConf.load(output / "resolved_config.yaml")
    from rlinf.scheduler.cluster.config import ClusterConfig

    validated = ClusterConfig.from_dict_cfg(resolved.cluster)
    validated_env = validated.node_groups[0].env_configs[0]
    node_env = resolved.cluster.node_groups[0].env_configs[0]
    if allocation_smoke:
        expected_vars = [{"EXISTING_ENV": "preserved"}] + [
            {key: value} for key, value in runtime_flags.items()
        ]
        assert node_env.env_vars == expected_vars
        assert validated_env.env_vars == expected_vars
        assert manifest["runtime_env_vars"] == runtime_flags
    else:
        assert "env_vars" not in node_env
        assert validated_env.env_vars is None
        assert "runtime_env_vars" not in manifest
    assert resolved.actor.global_batch_size == (global_batch_size or 64)
    assert manifest["batch_recipe"] == {
        "global_batch_size": resolved.actor.global_batch_size,
        "micro_batch_size": 1,
        "microbatches_per_rank": resolved.actor.global_batch_size // 4,
    }
    assert (
        resolved.actor.fsdp_config.enable_gradient_accumulation
        is not sync_every_microbatch
    )
    assert manifest["gradient_synchronization"] == {
        "sync_every_microbatch": sync_every_microbatch,
        "optimizer_accumulation_steps": resolved.actor.global_batch_size
        // (4 * resolved.actor.micro_batch_size),
    }
    assert manifest["actor_cuda_cache"] == {
        "retention_gib": cache_sizes[0],
        "preallocate_gib": cache_sizes[1],
    }
    assert manifest["pytorch_cuda_alloc_conf"] == "expandable_segments:True"
    if cache_sizes[0]:
        assert resolved.actor.cuda_cache_retention_gib == cache_sizes[0]
        assert resolved.actor.cuda_cache_preallocate_gib == cache_sizes[1]
    else:
        assert "cuda_cache_retention_gib" not in resolved.actor
        assert "cuda_cache_preallocate_gib" not in resolved.actor
    assert manifest["effective_learning_rate"] == resolved.actor.optim.lr
    assert manifest["effective_value_learning_rate"] == resolved.actor.optim.value_lr
    if learning_rate is not None:
        assert resolved.actor.optim.lr == resolved.actor.optim.value_lr == learning_rate
    if allocation_smoke:
        assert manifest["effective_seeds"] == dict.fromkeys(
            (
                "actor.seed",
                "rollout.seed",
                "env.train.seed",
                "runner.rollout_allocation.seed",
            ),
            2,
        )
    else:
        assert manifest["effective_seeds"]["actor.seed"] == 1234
        assert manifest["effective_seeds"]["env.train.seed"] == 1234
        assert manifest["effective_seeds"]["rollout.seed"] is None
    assert manifest["allocation_smoke"] is allocation_smoke
    assert manifest["allocation_smoke_overrides"] == (
        launcher.ALLOCATION_SMOKE_OVERRIDES if allocation_smoke else {}
    )
    assert (output / "resolved_config.yaml").is_file()
    if run:
        assert len(calls) == 2
        assert checks == [(4, "30720")]
        assert cleanups == [manifest["owner"]]
        expected = "FAILED" if fail_driver else "OK"
        assert (output / "status").read_text().strip() == expected
        assert all("stop" not in command for command, _ in calls)
        assert all(
            env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
            for _, env in calls
        )
    else:
        assert calls == cleanups == checks == []
        assert (output / "status").read_text().strip() == "DRY_RUN"


def test_duplicate_gpu_ids_rejected_before_launch(tmp_path):
    args = SimpleNamespace(gpus=[0, 0, 1, 2], updates=2, rollout_steps=32)
    with pytest.raises(ValueError, match="distinct"):
        launcher.build_config(args, {})


@pytest.mark.parametrize(
    "retention,preallocate",
    [
        (-1, 0),
        (28, -1),
        (0, 1),
        (28, 29),
        (float("inf"), 0),
        (28, float("nan")),
        (True, 0),
        (28, "26"),
    ],
)
def test_invalid_actor_cache_sizes_rejected_before_launch(retention, preallocate):
    args = SimpleNamespace(
        gpus=[0, 1, 2, 3],
        updates=2,
        rollout_steps=32,
        actor_cache_retention_gib=retention,
        actor_cache_preallocate_gib=preallocate,
    )
    with pytest.raises(ValueError, match="Actor cache sizes"):
        launcher.build_config(args, {})


def test_actor_cache_override_only_changes_allocator_config(tmp_path):
    from omegaconf import OmegaConf

    python, models = fixture_paths(tmp_path)
    args = SimpleNamespace(
        python=python,
        output=tmp_path / "out",
        gpus=[0, 1, 2, 3],
        updates=40,
        rollout_steps=520,
        config_name="libero_mopd_candidate_m1",
    )
    paths = {
        "student": models / "student",
        "libero_spatial": models / "spatial",
        "libero_object": models / "object",
        "libero_goal": models / "goal",
        "libero_10": models / "long",
    }
    original = OmegaConf.to_container(launcher.build_config(args, paths), resolve=True)
    args.actor_cache_retention_gib = 28
    args.actor_cache_preallocate_gib = 24
    changed = OmegaConf.to_container(launcher.build_config(args, paths), resolve=True)
    assert changed["actor"].pop("cuda_cache_retention_gib") == 28
    assert changed["actor"].pop("cuda_cache_preallocate_gib") == 24
    assert changed == original


def test_sync_every_microbatch_preserves_optimizer_batch(tmp_path):
    from omegaconf import OmegaConf

    python, models = fixture_paths(tmp_path)
    args = SimpleNamespace(
        python=python,
        output=tmp_path / "out",
        gpus=[0, 1, 2, 3],
        updates=40,
        rollout_steps=520,
        config_name="libero_mopd_candidate_m1",
    )
    paths = {
        "student": models / "student",
        "libero_spatial": models / "spatial",
        "libero_object": models / "object",
        "libero_goal": models / "goal",
        "libero_10": models / "long",
    }
    original = OmegaConf.to_container(launcher.build_config(args, paths), resolve=True)
    args.sync_every_microbatch = True
    changed = OmegaConf.to_container(launcher.build_config(args, paths), resolve=True)
    assert changed["actor"]["global_batch_size"] == 64
    assert changed["actor"]["micro_batch_size"] == 1
    assert changed["actor"]["fsdp_config"]["enable_gradient_accumulation"] is False
    changed["actor"]["fsdp_config"]["enable_gradient_accumulation"] = True
    assert changed == original


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


@pytest.mark.parametrize("arm,reuse", [("m0", 1), ("m1", 2)])
def test_candidate_launcher_preserves_quota_native_horizons(tmp_path, arm, reuse):
    python, models = fixture_paths(tmp_path)
    args = SimpleNamespace(
        python=python,
        output=tmp_path / "out",
        gpus=[0, 1, 2, 3],
        updates=2,
        rollout_steps=520,
        config_name=f"libero_mopd_candidate_{arm}",
    )
    paths = {
        "student": models / "student",
        "libero_spatial": models / "spatial",
        "libero_object": models / "object",
        "libero_goal": models / "goal",
        "libero_10": models / "long",
    }
    cfg = launcher.build_config(args, paths)
    assert cfg.algorithm.flow_refresh_updates == reuse
    assert cfg.env.train.total_num_envs == 32
    assert cfg.env.train.task_sampling_strategy == "suite_quota"
    args.rollout_steps = 32
    with pytest.raises(ValueError, match="native horizons"):
        launcher.build_config(args, paths)


def test_background_cuda_oom_aborts_live_child(tmp_path):
    import os
    import time

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="CUDA OOM"):
        launcher.run_logged(
            [
                sys.executable,
                "-u",
                "-c",
                "import time; print('torch.OutOfMemoryError: CUDA out of memory', flush=True); time.sleep(30)",
            ],
            dict(os.environ),
            tmp_path / "oom.log",
            timeout_seconds=10,
        )
    assert time.monotonic() - started < 10


def test_hung_smoke_is_time_bounded(tmp_path):
    import os

    with pytest.raises(TimeoutError, match="exceeded"):
        launcher.run_logged(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            dict(os.environ),
            tmp_path / "hang.log",
            timeout_seconds=0.1,
        )


def test_checkpoint_interval_must_preserve_round_boundaries(tmp_path):
    python, models = fixture_paths(tmp_path)
    args = SimpleNamespace(
        python=python,
        output=tmp_path / "out",
        gpus=[0, 1, 2, 3],
        updates=40,
        rollout_steps=520,
        save_every_updates=20,
    )
    paths = {
        "student": models / "student",
        "libero_spatial": models / "spatial",
        "libero_object": models / "object",
        "libero_goal": models / "goal",
        "libero_10": models / "long",
    }
    cfg = launcher.build_config(args, paths)
    assert cfg.runner.max_steps == 40 and cfg.runner.save_interval == 20
    for invalid in (0, 1, 3, 42, -2):
        args.save_every_updates = invalid
        with pytest.raises(ValueError, match="complete-round"):
            launcher.build_config(args, paths)


def adaptive_smoke_fixture(tmp_path, monkeypatch, arm):
    python, models = fixture_paths(tmp_path)
    monkeypatch.setenv("OPD_LIBERO_TEACHER_REFERENCE", str(tmp_path / "reference.json"))
    args = SimpleNamespace(
        python=python,
        output=tmp_path / "out",
        gpus=[0, 1, 2, 3],
        updates=8 if arm == 2 else 4,
        rollout_steps=520,
        config_name=f"libero_mopd_candidate_m{arm}",
        allocation_smoke=True,
    )
    paths = {
        "student": models / "student",
        "libero_spatial": models / "spatial",
        "libero_object": models / "object",
        "libero_goal": models / "goal",
        "libero_10": models / "long",
    }
    return args, paths


@pytest.mark.parametrize("arm", [2, 3])
def test_adaptive_smoke_only_changes_controller_window(tmp_path, monkeypatch, arm):
    from omegaconf import OmegaConf

    from rlinf.utils.rollout_allocation import training_signature

    args, paths = adaptive_smoke_fixture(tmp_path, monkeypatch, arm)
    smoke = launcher.build_config(args, paths)
    args.allocation_smoke = False
    default = launcher.build_config(args, paths)
    assert training_signature(smoke) == training_signature(default)
    assert smoke.runner.rollout_allocation.max_quota_change == 3
    assert smoke.runner.rollout_allocation.epsilon == 0.01
    assert default.runner.rollout_allocation.warmup_rounds == 20
    assert default.runner.rollout_allocation.window_rounds == 20
    assert default.runner.rollout_allocation.min_task_episodes == 5
    assert default.runner.rollout_allocation.rho == 0.5
    for key, value in launcher.ALLOCATION_SMOKE_OVERRIDES.items():
        assert smoke.runner.rollout_allocation[key] == value
        smoke.runner.rollout_allocation[key] = default.runner.rollout_allocation[key]
    assert OmegaConf.to_container(smoke, resolve=True) == OmegaConf.to_container(
        default, resolve=True
    )


@pytest.mark.parametrize("fault", ["uniform", "truncated", "too_short"])
def test_invalid_allocation_smoke_rejected(tmp_path, monkeypatch, fault):
    args, paths = adaptive_smoke_fixture(tmp_path, monkeypatch, 2)
    if fault == "uniform":
        args.config_name = "libero_mopd_candidate_m1"
    elif fault == "truncated":
        args.rollout_steps = 32
    else:
        args.updates = 4
    with pytest.raises(ValueError, match="allocation-smoke"):
        launcher.build_config(args, paths)


def test_smoke_window_covers_all_tasks_before_first_decision(tmp_path, monkeypatch):
    from collections import Counter

    from rlinf.algorithms.suite_distillation import normalized_suite_loss_weights
    from rlinf.utils.rollout_allocation import RolloutAllocator

    args, paths = adaptive_smoke_fixture(tmp_path, monkeypatch, 2)
    cfg = launcher.build_config(args, paths)
    suites = list(cfg.env.train.task_suite_names)
    tasks = [
        {
            "task_id": i * 10 + t,
            "task": f"{suite}/{t}",
            "suite": suite,
            "horizon": int(cfg.env.train.task_suite_horizons[suite]),
        }
        for i, suite in enumerate(suites)
        for t in range(10)
    ]
    # Synthetic outcomes/reference are exclusively a CPU unit-test fixture.
    allocation = RolloutAllocator(
        tasks, 4, 8, 1, cfg.runner.rollout_allocation, dict.fromkeys(suites, 0.8)
    )
    for round_index in range(2):
        plan = allocation.plan()
        assert allocation.quotas == [8] * 4
        assert allocation.last_diagnostics["reason"] == "uniform_or_warmup"
        allocation.observe(
            [
                dict(
                    row,
                    success_once=row["suite"] == suites[0],
                    simulation_steps=1 if row["suite"] == suites[0] else row["horizon"],
                )
                for row in plan
            ]
        )
    plan = allocation.plan()
    assert allocation.last_diagnostics["decision"]
    assert allocation.last_diagnostics["reason"] == "gap"
    assert min(allocation.last_diagnostics["task_counts"].values()) >= 1
    assert len(allocation.last_diagnostics["task_counts"]) == 40
    assert allocation.quotas != [8] * 4
    assert sum(allocation.quotas) == 32 and min(allocation.quotas) >= 4
    assert all(abs(q - 8) <= 3 for q in allocation.quotas)
    counts = Counter(row["suite"] for row in plan)
    assert normalized_suite_loss_weights(counts, suites) == [
        q / 8 for q in allocation.quotas
    ]
    for worker in range(4):
        assert {r["suite"] for r in plan if r["worker"] == worker} == set(suites)


@pytest.mark.parametrize("arm", range(4))
@pytest.mark.parametrize("seed", [0, 1, 2, 2**31 - 1])
def test_explicit_seed_sets_all_four_candidate_streams(
    tmp_path, monkeypatch, arm, seed
):
    args, paths = adaptive_smoke_fixture(tmp_path, monkeypatch, arm)
    args.allocation_smoke = False
    args.seed = seed
    cfg = launcher.build_config(args, paths)
    assert (
        cfg.actor.seed
        == cfg.rollout.seed
        == cfg.env.train.seed
        == cfg.runner.rollout_allocation.seed
        == seed
    )


def test_omitted_seed_and_none_preserve_config_exactly(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    args, paths = adaptive_smoke_fixture(tmp_path, monkeypatch, 1)
    args.allocation_smoke = False
    original = launcher.build_config(args, paths)
    args.seed = None
    unchanged = launcher.build_config(args, paths)
    assert OmegaConf.to_container(original, resolve=True) == OmegaConf.to_container(
        unchanged, resolve=True
    )


@pytest.mark.parametrize("seed", [-1, 2**31, True, 1.0, "1"])
def test_invalid_seed_rejected_before_composition(tmp_path, seed):
    args = SimpleNamespace(gpus=[0, 1, 2, 3], rollout_steps=520, seed=seed)
    with pytest.raises(ValueError, match="seed must be an integer"):
        launcher.build_config(args, {})


def test_learning_rate_override_only_changes_two_optimizer_fields(
    tmp_path, monkeypatch
):
    from omegaconf import OmegaConf

    args, paths = adaptive_smoke_fixture(tmp_path, monkeypatch, 1)
    args.allocation_smoke = False
    original = launcher.build_config(args, paths)
    args.learning_rate = None
    unchanged = launcher.build_config(args, paths)
    assert OmegaConf.to_container(original, resolve=True) == OmegaConf.to_container(
        unchanged, resolve=True
    )
    args.learning_rate = 3e-5
    changed = launcher.build_config(args, paths)
    assert changed.actor.optim.lr == changed.actor.optim.value_lr == 3e-5
    changed.actor.optim.lr = original.actor.optim.lr
    changed.actor.optim.value_lr = original.actor.optim.value_lr
    assert OmegaConf.to_container(changed, resolve=True) == OmegaConf.to_container(
        original, resolve=True
    )


@pytest.mark.parametrize(
    "learning_rate", [0, -1e-5, float("nan"), float("inf"), -float("inf"), True, "3e-5"]
)
def test_invalid_learning_rate_rejected_before_composition(learning_rate):
    args = SimpleNamespace(
        gpus=[0, 1, 2, 3], rollout_steps=520, learning_rate=learning_rate
    )
    with pytest.raises(ValueError, match="learning-rate must be finite and positive"):
        launcher.build_config(args, {})


def test_canonical_profile_rejects_changed_learning_rate(tmp_path, monkeypatch):
    args, paths = adaptive_smoke_fixture(tmp_path, monkeypatch, 1)
    args.allocation_smoke = False
    args.student_padding_profile = "canonical-fixed570"
    args.learning_rate = 3e-5
    with pytest.raises(ValueError, match="expects M1 T32/R2/LR3e-6"):
        launcher.build_config(args, paths)


@pytest.mark.parametrize("global_batch_size", [16, 64, 256])
def test_native_global_batch_override_preserves_other_recipe_fields(
    tmp_path, monkeypatch, global_batch_size
):
    from omegaconf import OmegaConf

    args, paths = adaptive_smoke_fixture(tmp_path, monkeypatch, 1)
    args.allocation_smoke = False
    original = launcher.build_config(args, paths)
    args.global_batch_size = None
    unchanged = launcher.build_config(args, paths)
    assert OmegaConf.to_container(original, resolve=True) == OmegaConf.to_container(
        unchanged, resolve=True
    )
    assert original.actor.global_batch_size == 64
    args.global_batch_size = global_batch_size
    changed = launcher.build_config(args, paths)
    assert changed.actor.global_batch_size == global_batch_size
    assert changed.actor.micro_batch_size == 1
    changed.actor.global_batch_size = original.actor.global_batch_size
    assert OmegaConf.to_container(changed, resolve=True) == OmegaConf.to_container(
        original, resolve=True
    )


@pytest.mark.parametrize(
    "global_batch_size", [0, -16, 1, 4, 15, 17, 24, 64.0, True, "256", float("nan")]
)
def test_invalid_global_batch_rejected_before_composition(global_batch_size):
    args = SimpleNamespace(
        gpus=[0, 1, 2, 3], rollout_steps=520, global_batch_size=global_batch_size
    )
    with pytest.raises(ValueError, match="global-batch-size must be an integer"):
        launcher.build_config(args, {})


@pytest.mark.parametrize("global_batch_size", [64, 256])
def test_canonical_profile_rejects_explicit_global_batch(
    tmp_path, monkeypatch, global_batch_size
):
    args, paths = adaptive_smoke_fixture(tmp_path, monkeypatch, 1)
    args.allocation_smoke = False
    args.student_padding_profile = "canonical-fixed570"
    args.global_batch_size = global_batch_size
    with pytest.raises(ValueError, match="cannot be combined with canonical-fixed570"):
        launcher.build_config(args, paths)


@pytest.mark.parametrize(
    "runtime_flags",
    [
        {},
        {"OPD_LIBERO_AUTOMAP_EGL": "0"},
        {"OPD_LIBERO_AUTOMAP_EGL": "1"},
        {"NCCL_P2P_DISABLE": "0"},
        {"NCCL_IB_DISABLE": "1"},
        {
            "OPD_LIBERO_AUTOMAP_EGL": "1",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
        },
    ],
)
def test_runtime_wrapper_preserves_optional_flags(monkeypatch, runtime_flags):
    from pathlib import Path

    values = {"PYTHONPATH": "/fixture/repo", "LIBERO_CONFIG_PATH": "/fixture/libero"}
    values.update(runtime_flags)

    def check_output(command, *, text):
        assert command[:2] == ["/fixture/python", "-c"]
        for key in launcher.OPTIONAL_RUNTIME_FLAGS:
            assert key in command[2]
        assert text is True
        return json.dumps(values)

    monkeypatch.setattr(launcher.subprocess, "check_output", check_output)
    assert launcher.read_runtime_environment(Path("/fixture/python")) == values


@pytest.mark.parametrize("key", launcher.OPTIONAL_RUNTIME_FLAGS)
@pytest.mark.parametrize("value", ["", "true", "2", 1, None])
def test_runtime_wrapper_rejects_invalid_optional_flags(monkeypatch, key, value):
    from pathlib import Path

    values = {
        "PYTHONPATH": "/fixture/repo",
        "LIBERO_CONFIG_PATH": "/fixture/libero",
        key: value,
    }
    monkeypatch.setattr(
        launcher.subprocess, "check_output", lambda *args, **kwargs: json.dumps(values)
    )
    with pytest.raises(ValueError, match=f"{key} must be"):
        launcher.read_runtime_environment(Path("/fixture/python"))
