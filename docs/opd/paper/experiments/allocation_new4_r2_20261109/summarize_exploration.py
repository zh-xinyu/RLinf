"""Recompute development metrics from complete, matched raw episode records."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(__file__).resolve().parents[1]
TASKS = ("open_laptop", "handover_mic", "dump_bin_bigbin", "stack_bowls_two")
RUNS = {
    "uniform": ("action30_uniform_status.json", "5fzkk7q3", "Blackwell", 16),
    "sr": ("action30_sr_status.json", "d01hcdxd", "Blackwell", 16),
    "residual": ("action10_residual_status.json", "fnq9kq7u", "Ada", 8),
}


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    if not path.exists():
        return []
    # A writer may be appending the last record during this read.
    data = path.read_bytes()
    return [json.loads(line) for line in data.split(b"\n")[:-1] if line.strip()]


def evaluate_label(label, reference_scenes=None):
    directory = OUT / "development_action30" / label
    if not (directory / "summary.json").exists():
        return None, None
    summary = read_json(directory / "summary.json")
    scenes, scores, control_steps = {}, {}, 0
    for task in TASKS:
        raw = read_json(directory / task / "summary.json")
        episodes = raw["episodes"]
        assert raw["requested_episodes"] == raw["completed_episodes"] == len(episodes) == 20
        assert len({episode["pool_index"] for episode in episodes}) == 20
        successes = sum(bool(episode["success_once"]) for episode in episodes)
        assert successes == raw["successes"]
        scores[task] = successes / 20
        assert math.isclose(scores[task], raw["success_rate"], abs_tol=1e-12)
        assert math.isclose(scores[task], summary["task_sr"][task], abs_tol=1e-12)
        scenes[task] = [
            {key: episode.get(key) for key in (
                "pool_index", "reset_seed", "actual_reset_seed", "instruction", "actual_scene_info"
            )}
            for episode in sorted(episodes, key=lambda row: row["pool_index"])
        ]
        control_steps += sum(episode["simulation_steps"] for episode in episodes)
    assert control_steps == summary["control_steps"]
    assert math.isclose(sum(scores.values()) / 4, summary["macro_sr"], abs_tol=1e-12)
    if reference_scenes is not None:
        assert scenes == reference_scenes, f"Development scenes/prompts differ: {label}"
    return summary, scenes


def main():
    teachers, scenes = evaluate_label("teachers")
    assert teachers is not None
    initial, _ = evaluate_label("initial400", scenes)
    assert initial is not None
    labels = {"teachers": teachers, "initial400": initial}
    for method in RUNS:
        for round_number in (10, 30):
            label = f"{method}_round{round_number}"
            summary, _ = evaluate_label(label, scenes)
            if summary is not None:
                labels[label] = summary
    for label, summary in labels.items():
        gaps = {
            task: max(teachers["task_sr"][task] - summary["task_sr"][task], 0.0)
            for task in TASKS
        }
        summary["positive_teacher_gaps"] = gaps
        summary["max_teacher_gap"] = max(gaps.values())
        summary["all_within_delta_010"] = max(gaps.values()) <= 0.10 + 1e-12

    calibration = read_json(OUT / "calibration_action30" / "teacher_reference.json")
    # Actual calibration accounting comes from its execution status, not SR alone.
    calibration_status = read_json(OUT / "calibration_action30" / "status.json")
    methods = {}
    for method, (status_file, run_id, hardware, microbatch) in RUNS.items():
        status = read_json(OUT / status_file)
        directory = Path(status["run_directory"])
        updates = read_jsonl(directory / "optimizer_updates.jsonl")
        allocations = read_jsonl(directory / "rollout_allocation.jsonl")
        episodes = read_jsonl(directory / "rollout_episodes.jsonl")
        assert [row["update_step"] for row in updates] == list(range(1, len(updates) + 1))
        assert [row["rounds"] for row in allocations] == list(range(1, len(allocations) + 1))
        assert len(episodes) == 32 * len(allocations)
        assert sum(row["simulation_steps"] for row in episodes) == (
            allocations[-1]["simulation_steps"] if allocations else 0
        )
        for row in allocations:
            assert sum(row["quotas"].values()) == 32
        checkpoint_points = {}
        for round_number in (10, 30):
            label = f"{method}_round{round_number}"
            if label not in labels:
                continue
            cp = directory / directory.name / "checkpoints" / f"global_step_{2 * round_number}"
            assert (cp / "runner_state.json").exists()
            assert (cp / "actor/dcp_checkpoint/.metadata").exists()
            for task in TASKS:
                raw = read_json(OUT / "development_action30" / label / task / "summary.json")
                assert Path(raw["checkpoint"]).resolve() == cp.resolve()
            checkpoint_points[round_number] = {
                "training_trajectories": 32 * round_number,
                "training_control_steps": allocations[round_number - 1]["simulation_steps"],
            }
        curve = [(0, initial["max_teacher_gap"])] + [
            (round_number, labels[f"{method}_round{round_number}"]["max_teacher_gap"])
            for round_number in (10, 30) if f"{method}_round{round_number}" in labels
        ]
        complete = [point[0] for point in curve] == [0, 10, 30]
        auc = None
        if complete:
            auc = sum(
                (x1 - x0) * (y0 + y1) / 2
                for (x0, y0), (x1, y1) in zip(curve, curve[1:])
            ) / 30
        weighted_exposure = {
            task: sum(row[f"train/actor/suite_loss_weight/{task}"] * 64 for row in updates)
            for task in TASKS
        }
        methods[method] = {
            "status": status["status"], "run_id": run_id,
            "hardware": hardware, "microbatch_per_card": microbatch,
            "optimizer_updates": len(updates),
            "collected_rollout_rounds": len(allocations),
            "completed_training_rounds": len(updates) // 2,
            "training_trajectories": len(episodes),
            "training_control_steps": sum(row["simulation_steps"] for row in episodes),
            "quota_history": [{"round": row["rounds"], "quotas": row["quotas"]} for row in allocations],
            "raw_selected_query_count_per_task": {task: 64 * len(updates) for task in TASKS},
            "weighted_query_exposure_proxy_per_task": weighted_exposure,
            "development_checkpoints": checkpoint_points,
            "gap_curve": curve, "complete_curve": complete,
            "normalized_gap_auc_0_30": auc,
            "sustained_joint_attainment": any(
                labels[a]["all_within_delta_010"] and labels[b]["all_within_delta_010"]
                for a, b in zip(
                    ["initial400"] + [f"{method}_round{r}" for r in (10, 30) if f"{method}_round{r}" in labels],
                    [f"{method}_round{r}" for r in (10, 30) if f"{method}_round{r}" in labels],
                )
            ),
        }
        uniform_end = labels.get("uniform_round30")
        method_end = labels.get(f"{method}_round30")
        methods[method]["macro_endpoint_drop_over_5pp_vs_uniform"] = (
            None if uniform_end is None or method_end is None else
            method_end["macro_sr"] < uniform_end["macro_sr"] - .05 - 1e-12
        )

    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "COMPLETE" if all(m["complete_curve"] for m in methods.values()) else "INTERIM",
        "matched_development_scene_prompt_audit": "PASS",
        "development_episodes_per_task": 20, "training_seed_count": 1,
        "tasks": TASKS, "evaluation": labels, "methods": methods,
        "training_calibration_status": calibration_status,
        "training_calibration_reference": calibration,
        "limitations": [
            "Development selection data; the formal 100-scene pool is not used.",
            "SR and Uniform use Blackwell/micro16; MSE uses Ada/micro8. Cross-hardware differences are not causal allocation evidence.",
            "The MSE controller is uniform through round10. Its round10 differences cannot be attributed to allocation.",
            "Identical seeds do not ensure identical numerical prefixes, even on the same hardware.",
            "SR consumes an additional teacher calibration: 80 trajectories / 19335 control targets.",
            "Raw selected query counts are configured per optimizer update, not unique observations; weighted exposure is only a proxy.",
            "Failed/restarted attempt costs remain separately archived and are not merged into successful-run progress.",
            "A single seed and 20 episodes/task do not establish a robust ranking.",
        ],
    }
    (OUT / "results_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# 新四任务 R2 allocation：开发结果", "",
        f"状态：{result['phase']}；生成时间 {result['generated_utc']}。每任务20次，训练seed0。", "",
        "| 模型 / 轮次 | Open | Handover | Dump | Stack | Macro | Max gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in labels.items():
        values = [summary["task_sr"][task] for task in TASKS] + [summary["macro_sr"], summary["max_teacher_gap"]]
        lines.append(f"| {label} | " + " | ".join(f"{100 * value:.1f}%" for value in values) + " |")
    lines += ["", "| 方法 | 已完成更新 | 已采集轮次 | 训练轨迹 | 训练 control targets | G-AUC (0–30) |", "|---|---:|---:|---:|---:|---:|"]
    for method, values in methods.items():
        auc = values["normalized_gap_auc_0_30"]
        lines.append(f"| {method} | {values['optimizer_updates']} | {values['collected_rollout_rounds']} | {values['training_trajectories']} | {values['training_control_steps']} | " + ("待完成" if auc is None else f"{auc:.4f}") + " |")
    lines += ["", "SR校准额外成本：80轨迹 / 19335 control targets。开发评测成本另计，未并入训练交互。", "", "## 解读边界", ""]
    lines += [f"- {item}" for item in result["limitations"]]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"phase": result["phase"], "labels": list(labels), "scene_prompt_audit": "PASS", "methods": {
        method: {key: value[key] for key in ("status", "optimizer_updates", "collected_rollout_rounds", "normalized_gap_auc_0_30")}
        for method, value in methods.items()
    }}, indent=2))


if __name__ == "__main__":
    main()
