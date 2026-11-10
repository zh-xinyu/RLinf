# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""One-GPU four-suite real closed-loop equivalence probe for frozen-slot skipping."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.gr00t.gr00t_n1d5 import get_model
from toolkits.libero_gr00t.evaluate_full_checkpoint import candidate_model_config
from toolkits.libero_gr00t.probe import renderer_index


def digest(tensor: torch.Tensor) -> str:
    """Hash exact tensor bytes, including floating-point bit patterns."""
    value = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(renderer_index())
    from rlinf.envs.action_utils import prepare_actions
    from rlinf.envs.libero.libero_env import LiberoEnv

    torch.set_num_threads(1)
    cfg = OmegaConf.load(args.config)
    env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env.train, resolve=True))
    env_cfg.total_num_envs = 4
    env = LiberoEnv(
        env_cfg, num_envs=4, seed_offset=0, total_num_processes=1, worker_info=None
    )
    args.output.mkdir(parents=True)
    try:
        if torch.cuda.device_count() != 1 or torch.cuda.mem_get_info()[0] < 24 * 2**30:
            raise ValueError("Select one GPU with at least24GiB free")
        torch.cuda.set_per_process_memory_fraction(
            16 * 2**30 / torch.cuda.mem_get_info()[1]
        )
        model_cfg, _ = candidate_model_config(args.config)
        model_cfg.opd_native_rollout_batch_size = 1
        model_cfg.opd_skip_finished_rollout = True
        model = get_model(model_cfg).eval().requires_grad_(False).cuda()
        catalog = {row["task_id"]: row for row in env.rollout_task_catalog}
        plans = [
            dict(
                catalog[task],
                slot=slot,
                worker=0,
                epoch=0,
                round=1,
                occurrence=40,
                reset_seed=1234 + task,
                policy_update_step=0,
            )
            for slot, task in enumerate([0, 10, 20, 30])
        ]
        reference, variants = None, []
        for skip in [False, True]:
            env.set_rollout_tasks(plans)
            obs, _ = env.reset()
            active = torch.ones(4, dtype=torch.bool)
            model.opd_skip_finished_rollout = skip
            if not variants:
                model.predict_action_batch(dict(obs))
                torch.cuda.synchronize()
            torch.manual_seed(1234)
            rows, inference_seconds, live_forwards = [], 0.0, 0
            start = time.perf_counter()
            for decision in range(65):
                torch.cuda.synchronize()
                predict_start = time.perf_counter()
                actions, output = model.predict_action_batch(
                    dict(obs, opd_rollout_active_mask=active)
                )
                torch.cuda.synchronize()
                inference_seconds += time.perf_counter() - predict_start
                live_forwards += int(active.sum()) if skip else 4
                row = {
                    "decision": decision,
                    "active": active.tolist(),
                    "live_actions_sha256": digest(actions[active]),
                    "live_chains_sha256": digest(
                        output["forward_inputs"]["chains"][active]
                    ),
                    "cpu_rng_sha256": digest(torch.get_rng_state()),
                    "cuda_rng_sha256": digest(torch.cuda.get_rng_state()),
                }
                rows.append(row)
                if reference is not None and row != reference[decision]:
                    raise RuntimeError(
                        f"Closed-loop policy divergence at decision{decision}"
                    )
                if decision < 64:
                    commands = prepare_actions(
                        actions, "libero", "gr00t", 10, 7, env_cfg=env_cfg
                    )
                    observations, _, terminations, truncations, _ = env.chunk_step(
                        commands
                    )
                    obs = observations[-1]
                    active = ~(terminations | truncations)[:, -1]
            result = {
                "skip": skip,
                "wall_seconds": time.perf_counter() - start,
                "policy_seconds": inference_seconds,
                "actual_model_forwards": live_forwards,
                "episodes": env.rollout_episode_records(),
            }
            if reference is None:
                reference = rows
            elif result["episodes"] != variants[0]["episodes"]:
                raise RuntimeError(
                    "Closed-loop outcomes or physical-step counts differ"
                )
            variants.append(result)
            (
                args.output
                / ("optimized_steps.json" if skip else "baseline_steps.json")
            ).write_text(json.dumps(rows, indent=2) + "\n")
            (args.output / "progress.json").write_text(
                json.dumps(variants, indent=2) + "\n"
            )
            print(json.dumps(result), flush=True)
        report = {
            "passed": True,
            "variants": variants,
            "policy_speedup": variants[0]["policy_seconds"]
            / variants[1]["policy_seconds"],
            "rollout_speedup": variants[0]["wall_seconds"]
            / variants[1]["wall_seconds"],
            "scope": "Four paired tasks, one official state each; protocol equivalence, not SR estimate",
        }
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.env.close()


if __name__ == "__main__":
    main()
