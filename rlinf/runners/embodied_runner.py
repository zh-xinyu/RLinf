# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import queue
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Union

from omegaconf.dictconfig import DictConfig

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics, print_metrics_table
from rlinf.utils.rollout_allocation import RolloutAllocator, load_teacher_reference
from rlinf.utils.runner_utils import check_progress
from rlinf.utils.timers import Timer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: Union[
            "EmbodiedFSDPActor",
            "EmbodiedNFTFSDPPolicy",
            "EmbodiedSACFSDPPolicy",
            "AsyncEmbodiedSACFSDPPolicy",
        ],
        rollout: Union["MultiStepRolloutWorker", "AsyncMultiStepRolloutWorker"],
        env: Union["EnvWorker", "AsyncEnvWorker"],
        reward: Union["EmbodiedRewardWorker"] = None,
        critic=None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.critic = critic
        self.reward = reward
        self.weight_sync_interval = self.cfg.runner.weight_sync_interval
        self.overlap_env_bootstrap = bool(
            self.cfg.runner.get("overlap_env_bootstrap", False)
        )

        # Step-gated profiling: ``cluster.profiling.steps`` lists the global step
        profiling_raw = self.cfg.cluster.get("profiling", None)
        profiling_enabled = profiling_raw is not None and bool(
            profiling_raw.get("enabled", True)
        )
        profile_steps_raw = (
            profiling_raw.get("steps", None) if profiling_enabled else None
        )
        self._profile_all_steps = profiling_enabled and profile_steps_raw is None
        self._profile_steps: set[int] | None = (
            {int(s) for s in profile_steps_raw}
            if profile_steps_raw is not None
            else None
        )

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.actor_channel = Channel.create("Actor")
        if self.reward is not None:
            self.reward_channel = Channel.create("Reward")
        else:
            self.reward_channel = None

        # this timer checks if we should stop training
        self.run_timer = Timer(None)  # Timer that checks if we should stop training

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0
        self.optimizer_update_steps = (
            cfg.runner.get("step_unit", "rollout") == "optimizer_update"
        )
        self.updates_per_rollout = (
            int(cfg.algorithm.get("flow_refresh_updates", 1))
            if self.optimizer_update_steps
            else 1
        )
        self.cumulative_env_steps = 0.0
        self.rollout_allocator = None
        self.teacher_calibration_budget = {}

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.logger = get_logger()
        self.metric_logger = MetricLogger(cfg)
        self.enable_per_worker_metric_log = bool(
            self.cfg.runner.get("per_worker_log", False)
        )

        # Async logging setup
        self.stop_logging = False
        self.log_queue = queue.Queue()
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()

    def _log_worker(self):
        """Background thread for processing log messages."""
        while not self.stop_logging:
            try:
                # Wait for log message with timeout
                log_func, args = self.log_queue.get(timeout=0.1)
                log_func(*args)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Logging error: {e}")
                continue

    def print_metrics_table_async(
        self,
        step: int,
        total_steps: int,
        start_time: float,
        metrics: dict,
        start_step: int = 0,
    ):
        """Async version that puts table printing in queue."""
        self.log_queue.put(
            (
                print_metrics_table,
                (
                    step,
                    total_steps,
                    start_time,
                    metrics,
                    start_step,
                    self.metric_logger.log_path,
                ),
            )
        )

    def init_workers(self):
        # create worker in order to decrease the maximum memory usage
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()

        if self.reward is not None:
            self.reward.init_worker().wait()

        rollout_handle.wait()
        env_handle.wait()
        self.actor.init_worker().wait()
        allocation = self.cfg.runner.get("rollout_allocation", {})
        if allocation.get("enabled", False):
            layouts = self.env.describe_rollout_allocation().wait()
            first = layouts[0]
            if any(
                {k: v for k, v in x.items() if k != "worker"}
                != {k: v for k, v in first.items() if k != "worker"}
                for x in layouts
            ):
                raise ValueError(
                    "Rollout allocation needs homogeneous workers/task catalogs."
                )
            reference = None
            if allocation.get("mode", "adaptive") == "adaptive":
                reference = load_teacher_reference(
                    allocation.teacher_reference_file, self.cfg, first["tasks"]
                )
                with open(allocation.teacher_reference_file) as stream:
                    calibration = json.load(stream)
                self.teacher_calibration_budget = {
                    "trajectories": sum(t["episodes"] for t in calibration["tasks"]),
                    "simulation_steps": sum(
                        t["simulation_steps"] for t in calibration["tasks"]
                    ),
                    "wall_seconds": calibration["wall_seconds"],
                }
                with open(
                    os.path.join(self.metric_logger.log_path, "teacher_reference.json"),
                    "w",
                ) as stream:
                    json.dump(calibration, stream, indent=2)
            self.rollout_allocator = RolloutAllocator(
                first["tasks"],
                len(layouts),
                first["envs"],
                first["epochs"],
                allocation,
                reference,
            )

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        self.logger.info(f"Resuming training from checkpoint directory {resume_dir}.")
        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        if self.optimizer_update_steps:
            with open(os.path.join(resume_dir, "runner_state.json")) as stream:
                state = json.load(stream)
            with open(
                os.path.join(resume_dir, "rollout_allocation_state.json")
            ) as stream:
                allocation_state = json.load(stream)
            step = state["optimizer_updates"]
            if (
                state["step_unit"] != "optimizer_update"
                or step <= 0
                or step >= self.max_steps
                or step % self.updates_per_rollout
                or state["rollout_count"] != step // self.updates_per_rollout
                or state["rollout_count"] != allocation_state["rounds"]
                or state["env_steps"] != allocation_state["simulation_steps"]
            ):
                raise ValueError(
                    "Inconsistent runner/allocation checkpoint or changed R."
                )
            self.rollout_allocator.load_state_dict(allocation_state)
            self.actor.load_checkpoint(actor_checkpoint_path).wait()
            actor_states = self.actor.restore_opd_update_counters(step).wait()
            self.global_step = step
            self.cumulative_env_steps = state["env_steps"]
            with open(
                os.path.join(self.metric_logger.log_path, "resume_metadata.json"), "w"
            ) as stream:
                json.dump(
                    {
                        "mode": "adaptive_round_boundary",
                        "source": str(resume_dir),
                        "runner_state": state,
                        "actor_states": actor_states,
                        "recovery_interactions": dict(
                            self.cfg.runner.get("recovery_interactions", {})
                        ),
                        "exact_replay": False,
                        "note": "Restores optimizer, scheduler, allocation history and task cursors; rollout proposal RNG restarts.",
                    },
                    stream,
                    indent=2,
                )
            self.logger.info(
                f"Restored adaptive rollout boundary at update {step}; rollout RNG restarts."
            )
        else:
            self.actor.load_checkpoint(actor_checkpoint_path).wait()
            self.global_step = int(resume_dir.split("global_step_")[-1])

    def update_rollout_weights(self):
        rollout_handle: Handle = self.rollout.sync_model_from_actor()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def _log_ranked_metrics(
        self,
        metrics_list: list[dict] | None,
        step: int,
        prefix: str,
        worker_group_name: str,
        add_prefix: bool = True,
    ):
        if not self.enable_per_worker_metric_log or not metrics_list:
            return
        for rank, metrics in enumerate(metrics_list):
            if not metrics:
                continue
            metrics_to_log = (
                {f"{prefix}/{k}": v for k, v in metrics.items()}
                if add_prefix
                else metrics
            )
            self.metric_logger.log(
                data=metrics_to_log,
                step=step,
                worker_group_name=worker_group_name,
                rank=rank,
            )

    def _aggregate_numeric_metrics(self, metrics_list: list[dict] | None) -> dict:
        if not metrics_list:
            return {}
        merged_metrics = defaultdict(list)
        for metrics in metrics_list:
            if not metrics:
                continue
            for key, value in metrics.items():
                merged_metrics[key].append(value)
        return {
            key: (sum(values) / len(values))
            for key, values in merged_metrics.items()
            if values
        }

    def _process_ranked_numeric_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = self._aggregate_numeric_metrics(metric_list)
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = self._aggregate_numeric_metrics(
                    metrics_list
                )
        return aggregated_metrics, ranked_metrics_list

    def _process_ranked_eval_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = (
            compute_evaluate_metrics(metric_list) if metric_list else {}
        )
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = compute_evaluate_metrics(metrics_list)
        return aggregated_metrics, ranked_metrics_list

    @staticmethod
    def _split_pipeline_actor_results(
        results: list[dict] | None,
    ) -> tuple[list[dict], list[dict]]:
        if not results:
            return [], []
        rollout_metrics = [result.get("rollout_metrics", {}) for result in results]
        training_metrics = [result.get("training_metrics", {}) for result in results]
        return rollout_metrics, training_metrics

    def _maybe_eval_and_checkpoint(self, step: int) -> dict:
        run_val, save_model, _ = check_progress(
            self.global_step,
            self.max_steps,
            self.cfg.runner.val_check_interval,
            self.cfg.runner.save_interval,
            1.0,
            run_time_exceeded=False,
        )

        eval_metrics = {}
        if run_val:
            with self.timer("eval"):
                self.update_rollout_weights()
                eval_metrics = self.evaluate()
                eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                self.metric_logger.log(data=eval_metrics, step=step)

        if save_model:
            self._save_checkpoint()

        return eval_metrics

    def _log_step_metrics(
        self,
        step: int,
        start_time: float,
        start_step: int,
        env_handle: Handle,
        rollout_handle: Handle,
        actor_training_handle: Handle,
        reward_handle: Handle | None,
        actor_rollout_metrics: list[dict],
        actor_training_metrics: list[dict],
        eval_metrics: dict,
    ) -> None:
        time_metrics = self.timer.consume_durations()
        time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
        if self.optimizer_update_steps and "time/step" in time_metrics:
            time_metrics["time/rollout_round"] = time_metrics["time/step"]
            time_metrics["time/amortized_update"] = (
                time_metrics["time/step"] / self.updates_per_rollout
            )
        env_time_metrics, env_time_metrics_per_rank = env_handle.consume_durations(
            return_per_rank=True
        )
        rollout_time_metrics, rollout_time_metrics_per_rank = (
            rollout_handle.consume_durations(return_per_rank=True)
        )
        actor_time_metrics, actor_time_metrics_per_rank = (
            actor_training_handle.consume_durations(return_per_rank=True)
        )
        time_metrics.update({f"time/env/{k}": v for k, v in env_time_metrics.items()})
        time_metrics.update(
            {f"time/rollout/{k}": v for k, v in rollout_time_metrics.items()}
        )
        time_metrics.update(
            {f"time/actor/{k}": v for k, v in actor_time_metrics.items()}
        )
        if self.reward is not None:
            assert reward_handle is not None
            reward_time_metrics, reward_time_metrics_per_rank = (
                reward_handle.consume_durations(return_per_rank=True)
            )
            time_metrics.update(
                {f"time/reward/{k}": v for k, v in reward_time_metrics.items()}
            )

        env_results = env_handle.wait()
        env_results_list = [results for results in env_results if results is not None]
        env_metrics = compute_evaluate_metrics(env_results_list)
        env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
        ranked_env_results = [
            {"rank": rank, "env": rank_metrics}
            for rank, rank_metrics in enumerate(env_results)
            if rank_metrics is not None
        ]
        _, env_metrics_per_rank = self._process_ranked_eval_results(
            ranked_env_results, metric_field="env"
        )

        rollout_metrics = {
            f"rollout/{k}": v
            for k, v in self._aggregate_numeric_metrics(actor_rollout_metrics).items()
        }
        training_metrics = {
            f"train/{k}": v
            for k, v in self._aggregate_numeric_metrics(actor_training_metrics).items()
        }

        self.metric_logger.log(env_metrics, step)
        self.metric_logger.log(rollout_metrics, step)
        self.metric_logger.log(time_metrics, step)
        if not self.optimizer_update_steps:
            self.metric_logger.log(training_metrics, step)
        else:
            # Do not overwrite the last individual update with a round average.
            training_metrics = {
                f"round/{key.removeprefix('train/')}": value
                for key, value in training_metrics.items()
                if "/update_" not in key
            }
            self.metric_logger.log(training_metrics, step)
            self.metric_logger.log(
                {
                    "rollout_count": self.global_step // self.updates_per_rollout,
                    "rollout/policy_update_step": (
                        self.global_step - self.updates_per_rollout
                    ),
                    "interaction/env_steps": self.cumulative_env_steps,
                    **{
                        f"rollout_success/{key.removeprefix('env/')}": value
                        for key, value in env_metrics.items()
                        if "success" in key
                    },
                },
                step,
            )
        self._log_ranked_metrics(
            metrics_list=actor_rollout_metrics,
            step=step,
            prefix="rollout",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=actor_training_metrics,
            step=step,
            prefix="train",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=actor_time_metrics_per_rank,
            step=step,
            prefix="time/actor",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=rollout_time_metrics_per_rank,
            step=step,
            prefix="time/rollout",
            worker_group_name=self.rollout.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=env_time_metrics_per_rank,
            step=step,
            prefix="time/env",
            worker_group_name=self.env.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=env_metrics_per_rank,
            step=step,
            prefix="env",
            worker_group_name=self.env.worker_group_name,
        )
        if self.reward is not None:
            self._log_ranked_metrics(
                metrics_list=reward_time_metrics_per_rank,
                step=step,
                prefix="time/reward",
                worker_group_name=self.reward.worker_group_name,
            )

        logging_metrics = time_metrics
        logging_metrics.update(eval_metrics)
        logging_metrics.update(env_metrics)
        logging_metrics.update(rollout_metrics)
        logging_metrics.update(training_metrics)

        self.print_metrics_table_async(
            step, self.max_steps, start_time, logging_metrics, start_step
        )

    def _finish_run(self) -> None:
        self.metric_logger.finish()

        # Stop logging thread
        self.stop_logging = True
        self.log_queue.join()  # Wait for all queued logs to be processed
        self.log_thread.join(timeout=1.0)

    def _should_profile_step(self, step_idx: int) -> bool:
        return self._profile_all_steps or (
            self._profile_steps is not None and step_idx in self._profile_steps
        )

    def _open_profiling_window(self, step_idx: int) -> None:
        """Dispatch ``start_profile`` to all compute worker groups for this step."""
        self.logger.info(f"Opening profiling window at step {step_idx}")
        self.actor.start_profile(step_idx).wait()
        self.rollout.start_profile(step_idx).wait()
        self.env.start_profile(step_idx).wait()

    def _close_profiling_window(self, step_idx: int) -> None:
        """Dispatch ``stop_profile`` to all compute worker groups."""
        self.actor.stop_profile().wait()
        self.rollout.stop_profile().wait()
        self.env.stop_profile().wait()
        self.logger.info(f"Closed profiling window at step {step_idx}")

    def run(self):
        if self.cfg.runner.get("use_training_pipeline", False):
            return self.run_pipeline()

        start_step = self.global_step
        start_time = time.time()
        for _step in range(start_step, self.max_steps, self.updates_per_rollout):
            # set global step
            self.actor.set_global_step(
                self.global_step // self.updates_per_rollout
                if self.optimizer_update_steps
                else self.global_step
            )
            self.rollout.set_global_step(self.global_step)

            profiled_step = (
                self.global_step
                if self._should_profile_step(self.global_step)
                else None
            )
            if profiled_step is not None:
                self._open_profiling_window(profiled_step)

            with self.timer("step", trace_args={"step_idx": _step}):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()
                with self.timer("generate_rollouts"):
                    if getattr(self, "rollout_allocator", None) is not None:
                        plan = self.rollout_allocator.plan()
                        self.env.set_rollout_allocation(plan, self.global_step).wait()
                    env_handle: Handle = self.env.interact(
                        input_channel=self.env_channel,
                        rollout_channel=self.rollout_channel,
                        reward_channel=self.reward_channel,
                        actor_channel=self.actor_channel,
                    )
                    rollout_handle: Handle = self.rollout.generate(
                        input_channel=self.rollout_channel,
                        output_channel=self.env_channel,
                    )
                    reward_handle = None
                    if self.reward is not None:
                        reward_handle: Handle = self.reward.compute_rewards(
                            input_channel=self.reward_channel,
                            output_channel=self.env_channel,
                        )
                    self.actor.recv_rollout_trajectories(
                        input_channel=self.actor_channel
                    ).wait()
                    rollout_handle.wait()
                    if self.reward is not None:
                        reward_handle.wait()

                if getattr(self, "rollout_allocator", None) is not None:
                    env_handle.wait()
                    self._record_allocation_round()

                # compute advantages and returns.
                with self.timer("cal_adv_and_returns"):
                    actor_rollout_metrics = (
                        self.actor.compute_advantages_and_returns().wait()
                    )

                # actor training.
                with self.timer("actor_training"):
                    actor_training_handle: Handle = self.actor.run_training()
                    env_bootstrap_handle: Handle | None = None
                    if (
                        self.overlap_env_bootstrap
                        and _step + self.updates_per_rollout < self.max_steps
                    ):
                        env_bootstrap_handle = self.env.prefetch_train_bootstrap(
                            rollout_channel=self.rollout_channel
                        )

                    actor_training_metrics = actor_training_handle.wait()
                    if env_bootstrap_handle is not None:
                        env_bootstrap_handle.wait()

                if self.optimizer_update_steps:
                    self._log_optimizer_updates(actor_training_metrics)
                self.global_step += self.updates_per_rollout
                log_step = self.global_step if self.optimizer_update_steps else _step
                eval_metrics = self._maybe_eval_and_checkpoint(log_step)

            if profiled_step is not None:
                self._close_profiling_window(profiled_step)

            self._log_step_metrics(
                step=log_step,
                start_time=start_time,
                start_step=start_step,
                env_handle=env_handle,
                rollout_handle=rollout_handle,
                actor_training_handle=actor_training_handle,
                reward_handle=reward_handle,
                actor_rollout_metrics=actor_rollout_metrics,
                actor_training_metrics=actor_training_metrics,
                eval_metrics=eval_metrics,
            )

        self._finish_run()

    def _log_optimizer_updates(self, actor_metrics: list[dict]) -> None:
        """Log each completed update without averaging over the reused pool."""
        metrics = self._aggregate_numeric_metrics(actor_metrics)
        repeats = self.updates_per_rollout
        expected = self.global_step + repeats
        if int(metrics["refresh/optimizer_updates"]) != expected:
            raise RuntimeError(
                f"Expected {expected} optimizer updates, got "
                f"{metrics['refresh/optimizer_updates']}."
            )
        mask_steps = metrics["refresh/real_env_steps_this_rollout"]
        if getattr(self, "rollout_allocator", None) is not None:
            recorded_steps = sum(
                r["simulation_steps"] for r in self.rollout_allocator.history[-1]
            )
            if recorded_steps != int(mask_steps):
                raise RuntimeError(
                    f"Environment/mask step accounting mismatch: {recorded_steps} != {mask_steps}"
                )
            self.cumulative_env_steps = self.rollout_allocator.simulation_steps
        else:
            self.cumulative_env_steps += mask_steps
        for inner in range(1, repeats + 1):
            prefix = f"refresh/update_{inner}/"
            row = {
                f"train/{key.removeprefix(prefix)}": value
                for key, value in metrics.items()
                if key.startswith(prefix)
            }
            row.update(
                {
                    "update_step": self.global_step + inner,
                    "rollout_count": self.global_step // repeats + 1,
                    "train/refresh/pool_update": inner,
                    "train/refresh/is_refreshed": int(inner > 1),
                    "interaction/env_steps": self.cumulative_env_steps,
                }
            )
            if getattr(self, "rollout_allocator", None) is not None:
                row.update(self._allocation_interaction_metrics())
                if inner == repeats:
                    row.update(self._last_allocation_metrics)
            if inner == 1:
                row["train/new_rollout_velocity_mse"] = row["train/actor/velocity_mse"]
                for key, value in list(row.items()):
                    if key.startswith("train/actor/velocity_mse/"):
                        row[
                            key.replace(
                                "actor/velocity_mse/", "new_rollout_velocity_mse/"
                            )
                        ] = value
            self.metric_logger.log(row, self.global_step + inner)
            with open(
                os.path.join(self.metric_logger.log_path, "optimizer_updates.jsonl"),
                "a",
            ) as stream:
                stream.write(json.dumps(row) + "\n")

    def _allocation_interaction_metrics(self) -> dict:
        allocator = self.rollout_allocator
        calibration = getattr(self, "teacher_calibration_budget", {})
        recovery = (
            getattr(self, "cfg", {}).get("runner", {}).get("recovery_interactions", {})
        )
        return {
            "interaction/recovery_simulation_steps": recovery.get(
                "simulation_steps", 0
            ),
            "interaction/recovery_trajectories": recovery.get("trajectories", 0),
            "interaction/total_training_rollout_rounds": allocator.rounds
            + recovery.get("rollout_rounds", 0),
            "interaction/total_simulation_steps": allocator.simulation_steps
            + calibration.get("simulation_steps", 0)
            + recovery.get("simulation_steps", 0),
            "interaction/total_trajectories": allocator.trajectories
            + calibration.get("trajectories", 0)
            + recovery.get("trajectories", 0),
            "interaction/rollout_rounds": allocator.rounds,
            "interaction/rollout_trajectories": allocator.trajectories,
            "interaction/simulation_steps": allocator.simulation_steps,
            "interaction/control_steps": allocator.simulation_steps,
        }

    def _record_allocation_round(self) -> None:
        """Persist physical outcomes before they can be reused by actor updates."""
        records = [
            r
            for worker in self.env.get_rollout_allocation_records().wait()
            for r in worker
        ]
        self.rollout_allocator.observe(records)
        path = self.metric_logger.log_path
        with open(os.path.join(path, "rollout_episodes.jsonl"), "a") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
        state = self.rollout_allocator.state_dict()
        with open(os.path.join(path, "rollout_allocation.jsonl"), "a") as stream:
            stream.write(
                json.dumps({k: v for k, v in state.items() if k != "history"}) + "\n"
            )
        metrics = self._allocation_interaction_metrics()
        if self.rollout_allocator.rounds == 1:
            metrics.update(
                {
                    f"calibration/{key}": value
                    for key, value in getattr(
                        self, "teacher_calibration_budget", {}
                    ).items()
                }
            )
        metrics["rollout_count"] = self.rollout_allocator.rounds
        for suite, quota in state["quotas"].items():
            selected = [r for r in records if r["suite"] == suite]
            metrics[f"allocation/quota/{suite}"] = quota
            metrics[f"allocation/collected/{suite}"] = len(selected)
            metrics[f"allocation/simulation_steps/{suite}"] = sum(
                r["simulation_steps"] for r in selected
            )
            metrics[f"allocation/success_once/{suite}"] = sum(
                r["success_once"] for r in selected
            ) / len(selected)
        for name in ("gap", "success", "probabilities", "signed_progress"):
            for suite, value in state["diagnostics"].get(name, {}).items():
                metrics[f"allocation/{name}/{suite}"] = value
        # Emit only after the inner updates, keeping W&B steps monotonic.
        self._last_allocation_metrics = metrics

    def run_pipeline(self):
        start_step = self.global_step
        start_time = time.time()
        for _step in range(start_step, self.max_steps):
            # set global step
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)

            profiled_step = (
                self.global_step
                if self._should_profile_step(self.global_step)
                else None
            )
            if profiled_step is not None:
                self._open_profiling_window(profiled_step)

            with self.timer("step", trace_args={"step_idx": _step}):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()
                env_handle: Handle = self.env.interact(
                    input_channel=self.env_channel,
                    rollout_channel=self.rollout_channel,
                    reward_channel=self.reward_channel,
                    actor_channel=self.actor_channel,
                )
                rollout_handle: Handle = self.rollout.generate(
                    input_channel=self.rollout_channel,
                    output_channel=self.env_channel,
                )
                reward_handle = None
                if self.reward is not None:
                    reward_handle: Handle = self.reward.compute_rewards(
                        input_channel=self.reward_channel,
                        output_channel=self.env_channel,
                    )
                # actor training.
                actor_training_handle: Handle = self.actor.run_training(
                    input_channel=self.actor_channel
                )
                with self.timer("generate_rollouts"):
                    rollout_handle.wait()
                    if self.reward is not None:
                        reward_handle.wait()

                env_bootstrap_handle: Handle | None = None
                if self.overlap_env_bootstrap and _step + 1 < self.max_steps:
                    env_bootstrap_handle = self.env.prefetch_train_bootstrap(
                        rollout_channel=self.rollout_channel
                    )

                actor_results = actor_training_handle.wait()
                actor_rollout_metrics, actor_training_metrics = (
                    self._split_pipeline_actor_results(actor_results)
                )
                if env_bootstrap_handle is not None:
                    env_bootstrap_handle.wait()

                self.global_step += 1
                eval_metrics = self._maybe_eval_and_checkpoint(_step)

            if profiled_step is not None:
                self._close_profiling_window(profiled_step)

            self._log_step_metrics(
                step=_step,
                start_time=start_time,
                start_step=start_step,
                env_handle=env_handle,
                rollout_handle=rollout_handle,
                actor_training_handle=actor_training_handle,
                reward_handle=reward_handle,
                actor_rollout_metrics=actor_rollout_metrics,
                actor_training_metrics=actor_training_metrics,
                eval_metrics=eval_metrics,
            )

        self._finish_run()

    def _save_checkpoint(self):
        self.logger.info(f"Saving checkpoint at step {self.global_step}.")
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()
        if self.optimizer_update_steps:
            with open(
                os.path.join(base_output_dir, "runner_state.json"), "w"
            ) as stream:
                json.dump(
                    {
                        "step_unit": "optimizer_update",
                        "optimizer_updates": self.global_step,
                        "rollout_count": self.global_step // self.updates_per_rollout,
                        "env_steps": self.cumulative_env_steps,
                    },
                    stream,
                    indent=2,
                )

        if getattr(self, "rollout_allocator", None) is not None:
            with open(
                os.path.join(base_output_dir, "rollout_allocation_state.json"), "w"
            ) as stream:
                json.dump(self.rollout_allocator.state_dict(), stream, indent=2)

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
