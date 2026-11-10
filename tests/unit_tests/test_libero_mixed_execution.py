# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Mixed5/10 execution preserves physical budgets and equal-observation gradients."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.losses import compute_analytic_local_kl_loss
from rlinf.utils.execution_protocol import (
    mask_executed_prefix,
    rollout_decision_steps,
    suite_execution_lengths,
)
from rlinf.utils.metric_utils import compute_loss_mask


def protocol():
    return OmegaConf.create(
        {
            "env_type": "libero",
            "freeze_finished_episodes": True,
            "auto_reset": False,
            "ignore_terminations": False,
            "task_suite_names": list("abcd"),
            "task_suite_horizons": dict(zip("abcd", [240, 240, 320, 480], strict=True)),
            "task_suite_execution_lengths": dict(
                zip("abcd", [5, 5, 5, 10], strict=True)
            ),
            "max_steps_per_rollout_epoch": 480,
        }
    )


def test_shared_decision_count_covers_every_physical_horizon():
    env = protocol()
    assert rollout_decision_steps(env, SimpleNamespace(num_action_chunks=10)) == 64
    del env.task_suite_execution_lengths
    assert rollout_decision_steps(env, SimpleNamespace(num_action_chunks=8)) == 60


@pytest.mark.parametrize(
    "change", ["autoreset", "partial_suites", "wrong_width", "bad_prefix"]
)
def test_invalid_execution_contract_rejected(change):
    cfg = protocol()
    width = 10
    if change == "autoreset":
        cfg.auto_reset = True
    elif change == "partial_suites":
        del cfg.task_suite_execution_lengths.a
    elif change == "wrong_width":
        width = 16
    else:
        cfg.task_suite_execution_lengths.a = 0
    with pytest.raises(ValueError):
        suite_execution_lengths(cfg, width)


def test_prefix_reduction_gradients_equal_separate_suite_means():
    torch.manual_seed(12)
    student = torch.randn(8, 10, 7, requires_grad=True)
    teacher = torch.randn_like(student, requires_grad=True)
    ids = torch.arange(4).repeat_interleave(2)
    lengths = torch.tensor([5, 5, 5, 10])[ids]
    kl = torch.linspace(0.2, 2, 8)
    coefficients = torch.tensor([0.4, 0.8, 1.2, 1.6])[ids]
    loss, metrics = compute_analytic_local_kl_loss(
        student,
        teacher,
        kl,
        action_prefix_lengths=lengths,
        sample_weights=coefficients,
        teacher_ids=ids,
        num_teachers=4,
    )
    expected = (
        sum(
            coefficients[2 * s]
            * (
                (
                    student[2 * s : 2 * s + 2, :k]
                    - teacher.detach()[2 * s : 2 * s + 2, :k]
                ).square()
                * kl[2 * s : 2 * s + 2, None, None]
            ).mean()
            for s, k in enumerate([5, 5, 5, 10])
        )
        / 4
    )
    torch.testing.assert_close(loss, expected)
    actual_grad = torch.autograd.grad(loss, student, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, student)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grad[:6, 5:].count_nonzero() == 0
    assert teacher.grad is None
    assert metrics["actor/velocity_mse"] > 0


def test_native_duplicate_padding_does_not_change_prefix_denominator():
    student = torch.ones(3, 10, 7, requires_grad=True)
    teacher = torch.zeros_like(student)
    lengths = torch.tensor([5, 10, 5])
    loss, _ = compute_analytic_local_kl_loss(
        student,
        teacher,
        torch.ones(3),
        loss_mask=torch.tensor([[1], [1], [0]], dtype=torch.bool),
        action_prefix_lengths=lengths,
    )
    loss.backward()
    assert loss.item() == pytest.approx(1)
    assert student.grad[2].count_nonzero() == 0
    torch.testing.assert_close(student.grad[0].sum(), student.grad[1].sum())


def test_physics_mask_excludes_unexecuted_slots():
    mask = torch.ones(2, 4, 10, dtype=torch.bool)
    ids = torch.arange(4).expand(3, 4)
    actual = mask_executed_prefix(
        mask, ids, list("abcd"), dict(zip("abcd", [5, 5, 5, 10], strict=True))
    )
    assert actual.sum() == 50
    assert not actual[:, :3, 5:].any()


def test_real_frozen_chunk_step_and_masks_match_simulation_counts(monkeypatch):
    pytest.importorskip("libero")
    from test_libero_multi_suite import make_cfg, make_env

    from rlinf.envs.libero import libero_env

    cfg = make_cfg(
        env_type="libero",
        task_sampling_strategy="suite_balanced",
        freeze_finished_episodes=True,
        reset_settle_steps=1,
        max_steps_per_rollout_epoch=20,
        max_episode_steps=20,
        task_suite_horizons=dict(zip("abcd", [10, 10, 10, 20], strict=True)),
        task_suite_execution_lengths=dict(zip("abcd", [5, 5, 5, 10], strict=True)),
    )
    env = make_env(libero_env, monkeypatch, cfg)
    steps = np.zeros(4, dtype=int)

    def step(actions, ids):
        ids = np.asarray(ids)
        steps[ids] += 1
        return (
            [{"tick": int(steps[i])} for i in ids],
            None,
            np.zeros(len(ids), dtype=bool),
            [{} for _ in ids],
        )

    env.env = SimpleNamespace(step=step)
    monkeypatch.setattr(
        env, "_wrap_obs", lambda raw: {"ticks": [x["tick"] for x in raw]}
    )
    monkeypatch.setattr(env, "_reconfigure", lambda *args: None)
    monkeypatch.setattr(env, "_calc_step_reward", lambda terms: terms.astype(float))
    env.reset()
    steps[:] = 0
    labels = [env._task_suite_labels[int(task)] for task in env.task_ids]
    lengths = np.array([cfg.task_suite_execution_lengths[s] for s in labels])
    obs1, _, term1, trunc1, _ = env.chunk_step(torch.zeros(4, 10, 7))
    np.testing.assert_array_equal(steps, lengths)
    assert obs1[-1]["ticks"] == lengths.tolist()
    _, _, term2, trunc2, _ = env.chunk_step(torch.zeros(4, 10, 7))
    np.testing.assert_array_equal(steps, lengths * 2)
    env.chunk_step(torch.zeros(4, 10, 7))
    np.testing.assert_array_equal(steps, lengths * 2)
    dones = torch.stack(
        [torch.zeros(4, 10, dtype=torch.bool), term1 | trunc1, term2 | trunc2]
    )
    mask, _ = compute_loss_mask(dones)
    ids = torch.tensor([list("abcd").index(s) for s in labels]).expand(3, 4)
    mask = mask_executed_prefix(
        mask, ids, list("abcd"), cfg.task_suite_execution_lengths
    )
    assert int(mask.sum()) == env.physics_step_counts["policy_simulation_steps"] == 50


def test_mixed_launcher_and_evaluator_contract(tmp_path):
    from test_libero_production_launcher import fixture_paths

    from toolkits.libero_gr00t.evaluate_full_checkpoint import candidate_model_config
    from toolkits.libero_gr00t.launch_production import build_config

    python, models = fixture_paths(tmp_path)
    paths = {
        s: models / name
        for s, name in zip(
            ["student", "libero_spatial", "libero_object", "libero_goal", "libero_10"],
            ["student", "spatial", "object", "goal", "long"],
            strict=True,
        )
    }
    args = SimpleNamespace(
        python=python,
        output=tmp_path / "plan",
        gpus=[0, 1, 2, 3],
        updates=4,
        rollout_steps=480,
        execution_protocol="official-mixed",
        config_name="libero_mopd_mixed_k_m1",
    )
    cfg = build_config(args, paths)
    assert rollout_decision_steps(cfg.env.train, cfg.actor.model) == 64
    assert cfg.actor.global_batch_size == 512
    assert cfg.actor.model.num_action_chunks == 10
    assert all(
        t.opd_rotary_precision == "bf16" for t in cfg.rollout.expert_models.values()
    )
    path = tmp_path / "resolved.yaml"
    OmegaConf.save(cfg, path)
    model, reuse = candidate_model_config(path)
    assert model.num_action_chunks == 10 and reuse == 2


@pytest.mark.parametrize(
    "mutation", [None, "duplicate_episode", "changed_weights", "wrong_prefix"]
)
def test_imported_teacher_reference_requires_real_coverage_and_identity(
    tmp_path, mutation
):
    import json

    from test_libero_production_launcher import fixture_paths

    from toolkits.libero_gr00t import import_official_teacher_reference as importer
    from toolkits.libero_gr00t.launch_production import build_config
    from toolkits.libero_gr00t.teacher_reference import checkpoint_identity, digest

    python, models = fixture_paths(tmp_path)
    names = ["student", "libero_spatial", "libero_object", "libero_goal", "libero_10"]
    paths = {
        s: models / name
        for s, name in zip(
            names, ["student", "spatial", "object", "goal", "long"], strict=True
        )
    }
    args = SimpleNamespace(
        python=python,
        output=tmp_path / "plan",
        gpus=[0, 1, 2, 3],
        updates=4,
        rollout_steps=480,
        execution_protocol="official-mixed",
        config_name="libero_mopd_mixed_k_m1",
    )
    cfg = build_config(args, paths)
    config = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config)
    capture = tmp_path / "capture"
    capture.mkdir()
    manifest = {
        "initial_states": list(range(20)),
        "cases": {},
        "teacher_files_sha256": {},
    }
    for suite, case in importer.CASES.items():
        (paths[suite] / "model.safetensors").write_bytes(
            b"unit-test checkpoint content"
        )
        manifest["teacher_files_sha256"][suite] = checkpoint_identity(paths[suite])
        manifest["cases"][case] = {"checkpoint": str(paths[suite].resolve())}
        folder = capture / case
        folder.mkdir()
        k = cfg.env.train.task_suite_execution_lengths[suite]
        horizon = cfg.env.train.task_suite_horizons[suite]
        (folder / "protocol.json").write_text(
            json.dumps({"execute": k, "horizon": horizon, "start_time": 100.0})
        )
        rows = [
            {
                "suite": suite,
                "task_id": t,
                "trial": r,
                "execute": k,
                "horizon": horizon,
                "steps": horizon,
                "complete_episode": True,
                "success_once": r % 2 == 0,
            }
            for t in range(10)
            for r in range(20)
        ]
        if suite == "libero_spatial":
            if mutation == "duplicate_episode":
                rows[1] = dict(rows[0])
            elif mutation == "wrong_prefix":
                rows[0]["execute"] = 10
        file = folder / "episodes.jsonl"
        file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (folder / "summary.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "episode_file_sha256": digest(file),
                    "finished_at": 200.0,
                }
            )
        )
    (capture / "manifest.json").write_text(json.dumps(manifest))
    (capture / "queue_status.json").write_text(json.dumps({"phase": "complete"}))
    if mutation == "changed_weights":
        (paths["libero_spatial"] / "model.safetensors").write_bytes(
            b"changed after calibration"
        )
    output = tmp_path / "reference.json"
    if mutation is None:
        result = importer.import_reference(config, capture, output)
        assert (
            len(result["tasks"]) == 40
            and result["calibration_budget"]["trajectories"] == 800
        )
    else:
        with pytest.raises(ValueError):
            importer.import_reference(config, capture, output)
        assert not output.exists()
