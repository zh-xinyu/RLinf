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

import copy
import json
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.expert import build_expert_model_config
from rlinf.algorithms.flow_opd import (
    build_flat_valid_transition_mask,
    count_flow_timesteps_by_suite,
    flow_teacher_transition_logprobs,
    map_target_flow_state_to_source,
    sample_flow_timestep_indices,
    select_flow_opd_logprobs,
    select_opd_metric_rows,
    teacher_suite_presence,
    transport_source_velocity_to_target,
)
from rlinf.algorithms.losses import flow_sde_kl_weights
from rlinf.algorithms.registry import (
    calculate_adv_and_returns,
    get_policy_loss,
    policy_loss,
)
from rlinf.algorithms.utils import (
    kl_penalty,
)
from rlinf.config import SupportedModel, torch_dtype_from_precision
from rlinf.data.embodied_io_struct import Trajectory, convert_trajectories_to_batch
from rlinf.data.io_struct import BatchResizingIterator, RolloutResult
from rlinf.data.lerobot_paths import resolve_lerobot_repo_id
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.fsdp.utils import (
    pack_fsdp_input,
    prepare_pack_fsdp,
    unpack_fsdp_logprobs,
    unpack_sequences,
)
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
    split_dynamic_batch_size,
)
from rlinf.utils.distributed import (
    RolloutDataBalance,
    all_reduce_dict,
    all_reduce_int,
    masked_normalization,
)
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    CRITIC_EXPLAINED_VARIANCE_KEY,
    append_to_dict,
    compute_critic_explained_variance_from_stats,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
    pop_critic_explained_variance_stats,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    cpu_weight_swap,
    get_loss_agg_func,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
)
from rlinf.workers.rollout.utils import RankMapper

# Losses of the flow-based OPD mechanism study that distill a frozen teacher's velocity
# field at student-visited flow states. They share one data path: no reward / advantage /
# critic, a post-rollout teacher velocity pass, and a supervised match against it.
# "velocity_distill" (M1) weights timesteps uniformly; "analytic_local_kl" (M3 and
# M3-Hybrid) weights them by the local Flow-SDE KL's w(tau).
VELOCITY_DISTILL_LOSS_TYPES = ("velocity_distill", "analytic_local_kl")

# Model families whose action head is a flow. They share the transition algebra that M3 and
# M4 are defined on, so adv_type="opd" scores a sampled *flow transition* for them rather
# than the action tokens the OpenVLA path scores.
FLOW_MODELS = (SupportedModel.OPENPI, SupportedModel.GR00T)


def process_nested_dict_for_adv(nested_dict, rollout_epoch):
    """
    original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
    target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_adv(value, rollout_epoch)
    return ret_dict


def process_nested_dict_for_train(nested_dict, shuffle_id):
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict


def trim_nested_tensor_time_dim(value, target_steps: int, key_path=()):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        assert value.shape[0] in {target_steps, target_steps + 1}, (
            f"Cannot trim field {'.'.join(key_path)!r} with shape "
            f"{tuple(value.shape)} to {target_steps} OPD training steps."
        )
        return value[:target_steps]
    if isinstance(value, dict):
        return {
            key: trim_nested_tensor_time_dim(
                nested_value, target_steps, (*key_path, key)
            )
            for key, nested_value in value.items()
        }
    raise TypeError(
        f"Unsupported field {'.'.join(key_path)!r} type {type(value)} for OPD trimming."
    )


def flatten_nested_tensor_time_batch(value, key_path=()):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        assert value.dim() >= 2, (
            f"Cannot flatten field {'.'.join(key_path)!r} with shape "
            f"{tuple(value.shape)} across time and batch."
        )
        return value.reshape(-1, *value.shape[2:])
    if isinstance(value, dict):
        return {
            key: flatten_nested_tensor_time_batch(nested_value, (*key_path, key))
            for key, nested_value in value.items()
        }
    raise TypeError(
        f"Unsupported field {'.'.join(key_path)!r} type {type(value)} for flattening."
    )


def compute_rollout_train_kl(
    m_batch: dict, loss_mask: torch.Tensor
) -> Optional[torch.Tensor]:
    """
    Compute the masked mean of absolute difference between rollout and training logprobs.

    Args:
        m_batch: Dictionary containing 'rollout_logprobs' and 'recomputed_logprobs'.
        loss_mask: Mask tensor for computing weighted mean.

    Returns:
        Masked mean of abs(recomputed_logprobs - rollout_logprobs), or None if keys are missing.
    """
    if "rollout_logprobs" not in m_batch or "recomputed_logprobs" not in m_batch:
        return None
    rollout_logprobs = m_batch["rollout_logprobs"]
    recomputed_logprobs = m_batch["recomputed_logprobs"]
    kl = torch.abs(recomputed_logprobs - rollout_logprobs)
    return masked_mean(kl, loss_mask)


class FSDPActor(FSDPModelManager, Worker):
    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        cfg_fsdp: Optional[DictConfig] = None,
    ) -> None:
        """
        FSDPActor worker used to train the model with data from rollout workers.

        Args:
            cfg (DictConfig): The global yaml configuration.
            placement (ModelParallelComponentPlacement): The accelerator placement for actor worker.
        """
        if cfg_fsdp is None:
            cfg_fsdp = cfg.actor
        Worker.__init__(self)
        super().__init__(cfg_fsdp, self._world_size, self._rank)

        self.cfg = cfg

        self.response_len = (
            cfg.actor.model.encoder_seq_length - cfg.data.max_prompt_length
        )
        self.calculate_entropy = cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.kl_beta = cfg.algorithm.kl_beta
        self.kl_penalty_type = cfg.algorithm.kl_penalty_type
        self.reinpp_kl_beta = cfg.algorithm.get("reinpp_kl_beta", 0.0)
        self.combine_reference_model = cfg.actor.get("combine_reference_model", True)

        self.total_batch_size_per_dp = (
            cfg.data.rollout_batch_size * cfg.algorithm.group_size // self._world_size
        )

        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = placement
        self.is_pipeline = self._component_placement.is_disaggregated
        self.ref_policy_state_dict = None
        if self.is_pipeline:
            self._inference_group_name = cfg.inference.group_name
            self._inference_world_size = self._component_placement.get_world_size(
                "inference"
            )
            self._inference_dst_map: dict[int, list[str]] = {}
        else:
            self._inference_group_name = None
            self._inference_world_size = 0
            self._inference_dst_map = None
        self.loss_agg_func = get_loss_agg_func(cfg.algorithm.loss_agg_func)
        self.enable_offload = not self.is_pipeline and cfg.actor.get(
            "enable_offload", False
        )
        self.micro_batch_size = cfg.actor.micro_batch_size
        self.n_mini_batches = cfg.algorithm.n_minibatches
        self.task_type = cfg.runner.task_type
        self.entropy_op_type = cfg.algorithm.get("entropy_op_type", "flash_attn")
        self.enable_dp_load_balance = cfg.actor.get("enable_dp_load_balance", False)
        self.lr_sched_sync_with_optim = cfg.actor.get("lr_sched_sync_with_optim", True)
        self.enable_dynamic_batch_size = cfg.runner.get(
            "enable_dynamic_batch_size", False
        )
        if self.is_pipeline:
            assert not self.enable_dp_load_balance, (
                "DP load balance is not supported in pipeline mode."
            )
            assert not self.enable_dynamic_batch_size, (
                "Dynamic batch size is not supported in pipeline mode."
            )
        self.max_tokens_per_mbs = cfg.runner.get("max_tokens_per_mbs", 2048)
        self.variable_seq_lengths = self.cfg.actor.model.get(
            "variable_seq_lengths", False
        )

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend
        (FSDP/FSDP2) to wrap it. If needed, offload model parameters and optimizer states to CPU.
        If kl_beta > 0, retrieve the reference policy model state dict to CPU.
        If mode is disaggregated, setup which inference ranks it needs to sync weights to by
        doing a handshake with inference workers.
        """
        self.setup_model_and_optimizer()
        if (
            self.kl_beta > 0 or self.reinpp_kl_beta > 0
        ) and self.combine_reference_model:
            self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model)
            self.offload_model_buffer = {}

        if self.enable_offload and not self.is_pipeline:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self._component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self) -> None:
        """Just for interface compatibility with MegatronActor."""
        pass

    def sync_model_to_inference(self) -> None:
        """
        Sync the model's full state dict to the inference worker.
        The model state_dict is the reference of actor's model
        parameters(by setting cpu_offload=False).
        """
        if not self._inference_dst_map:
            self._strategy.setup_actor_sync_inference_ranks(self)

        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device, False)

        inference_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        # NOTE: we have already know which inference rank needs which params
        # by calling _strategy.setup_actor_sync_inference_ranks() to do handshake
        # with each inference rank. just send them accordingly.
        for rank, needed_params in self._inference_dst_map.items():
            sended_params = {}
            for name in needed_params:
                if name in inference_state_dict:
                    # mentioned again, no ShardedTensor here.
                    sended_params[name] = (
                        inference_state_dict[name].to_local()
                        if isinstance(inference_state_dict[name], DTensor)
                        else inference_state_dict[name]
                    )
            self.send(
                object=sended_params,
                dst_group_name=self._inference_group_name,
                dst_rank=rank,
                async_op=True,
            )

        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

        torch.distributed.barrier()

    @Worker.timer("actor/sync_model_to_rollout")
    def sync_model_to_rollout(self):
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        rollout_dtype = None
        if self._cfg.get("sync_precision", None) is not None:
            rollout_dtype = torch_dtype_from_precision(self._cfg.sync_precision)

        rollout_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        has_visual = any("visual." in k for k in rollout_state_dict.keys())
        model_bucket_list = self.divide_model_to_bucket(rollout_state_dict, has_visual)
        del rollout_state_dict
        send_handles = []
        buffer = {}
        for bucket_idx, model_bucket in enumerate(model_bucket_list):
            for k, v in model_bucket.items():
                if isinstance(v, DTensor):
                    v = v.full_tensor()
                if rollout_dtype is not None:
                    v = v.to(rollout_dtype)
                if not self.is_pipeline:
                    v = reduce_tensor(v)
                buffer[k] = v
            if bucket_idx == 0:
                buffer["bucket_length"] = len(model_bucket_list)

            for send_handle in send_handles:
                send_handle.wait()
            send_handles = []

            if not self.is_pipeline:
                send_handle = self.send(
                    buffer,
                    self._rollout_group_name,
                    self._weight_dst_rank_in_rollout,
                    async_op=True,
                )
                send_handles.append(send_handle)
            else:
                for rank in self._weight_dst_rank_in_rollout:
                    send_handle = self.send(
                        buffer,
                        self._rollout_group_name,
                        rank,
                        async_op=True,
                    )
                    send_handles.append(send_handle)
            buffer = {}

        for send_handle in send_handles:
            send_handle.wait()

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad()

        clear_memory(sync=False)

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def get_dynamic_batch_as_much(
        self,
        input_channel: Channel,
        min_result_len: int,
        max_result_len: int,
        cliped_results=[],
        unfinished_result=None,
    ):
        assert not input_channel.is_local
        rollout_results = cliped_results
        # get min_result_len
        while len(rollout_results) < min_result_len:
            if unfinished_result is not None:
                rollout_result: RolloutResult = unfinished_result.wait()
                unfinished_result = None
            else:
                rollout_result: RolloutResult = input_channel.get()
            rollout_results.append(rollout_result)

        # try to get result as much
        # get result in every 0.1s and do all reduce to get the min result between dp (result_len)
        # stop at: the min result between dp (result_len) is same as the last min result
        last_result_len = 0
        result_len = len(rollout_results)
        time_until = time.time() + 0.1
        while last_result_len < result_len:
            if len(rollout_results) < max_result_len:
                if unfinished_result is None:
                    unfinished_result = input_channel.get(async_op=True)
                else:
                    time.sleep(0.001)
                if unfinished_result.done():
                    rollout_results.append(unfinished_result.wait())
                    unfinished_result = None
                if time.time() >= time_until:
                    last_result_len = result_len
                    result_len = all_reduce_int(len(rollout_results))
                    if last_result_len < result_len:
                        time_until = time.time() + 0.1
            else:
                last_result_len = result_len
                result_len = all_reduce_int(len(rollout_results))

        cliped_results = list(rollout_results[result_len:])
        rollout_results = rollout_results[:result_len]

        batches = []
        for rollout_result in rollout_results:
            batch = rollout_result.to_actor_batch(
                self.cfg.data.max_prompt_length,
                self.cfg.actor.model.encoder_seq_length,
                self.tokenizer.eos_token_id,
            )
            batches.append(batch)

        batch = RolloutResult.merge_batches(batches)
        rollout_result = RolloutResult.merge_result_list(rollout_results)
        return batch, rollout_result, result_len, cliped_results, unfinished_result

    @staticmethod
    def _split_to_micro_batch(
        batch,
        enable_dynamic_batch_size: bool,
        *,
        max_tokens_per_mbs: Optional[int] = None,
        split_num,
    ):
        if enable_dynamic_batch_size:
            (
                micro_batches_iter,
                _,
                micro_batch_cnt,
                dbs_indices,
            ) = split_dynamic_batch_size(
                batch=batch,
                cp_world_size=1,
                vpp_world_size=1,
                max_tokens_per_mbs=max_tokens_per_mbs,
                microbatch_group_size_per_vp_stage=1,
            )
        else:
            micro_batch_cnt = split_num
            micro_batches_iter = get_iterator_k_split(batch, micro_batch_cnt)
            dbs_indices = None
        return micro_batches_iter, micro_batch_cnt, dbs_indices

    def _load_weight_and_optimizer(self) -> None:
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        with self.device_lock:
            if not self.enable_offload:
                return
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

    def compute_logprobs(self, logits, target):
        return compute_logprobs_from_logits(
            logits,
            target,
            op_type=self.entropy_op_type,
        )

    def forward_batch(
        self, m_batch: dict[str, torch.Tensor], calculate_entropy: bool = False
    ) -> torch.Tensor:
        input_ids = m_batch["input_ids"]
        attention_mask = m_batch["attention_mask"]
        position_ids = m_batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in m_batch.keys():
            for key in m_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in m_batch["multi_modal_inputs"]],
                    dim=0,
                ).to(Worker.torch_device_type)

        if self.enable_dynamic_batch_size or self.variable_seq_lengths:
            max_seq_len_pack = self.max_tokens_per_mbs
            max_seq_len_unpack = self.cfg.actor.model.encoder_seq_length
            max_prompt_len = self.cfg.data.max_prompt_length
            max_response_len = max_seq_len_unpack - max_prompt_len
            idx_starts, idx_ends = prepare_pack_fsdp(m_batch, max_prompt_len)

            input_ids, position_ids, attention_mask = pack_fsdp_input(
                input_ids,
                position_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_pack=max_seq_len_pack,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_to_fixed_len=not self.variable_seq_lengths,
            )

        with self.amp_context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                **multi_modal_inputs,
            )

        logits: torch.Tensor = outputs.logits

        logits.div_(self.cfg.algorithm.sampling_params.temperature)

        if self.enable_dynamic_batch_size or self.variable_seq_lengths:
            logprobs = unpack_fsdp_logprobs(
                logits,
                input_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_unpack=max_seq_len_unpack,
                eos_token_id=self.tokenizer.eos_token_id,
                compute_logprobs_fn=self.compute_logprobs,
            )
            logprobs = logprobs[:, -max_response_len:]
        else:
            # (bsz, response_length, vocab_size)
            logits = logits[:, -self.response_len - 1 : -1, :]
            responses = input_ids[:, -self.response_len :]
            logprobs = self.compute_logprobs(logits, responses)

        if calculate_entropy:
            entropy = compute_entropy_from_logits(logits)

            if self.enable_dynamic_batch_size or self.variable_seq_lengths:
                entropy = unpack_sequences(
                    entropy, idx_starts, idx_ends, max_seq_len_unpack, pad_val=0
                )[:, -self.response_len :]

            return logprobs, entropy

        return logprobs

    def inference_step(
        self,
        batch: dict[str, torch.Tensor],
        num_sequences: int,
        compute_ref_logprobs: bool,
    ):
        micro_batches_iter, _, dbs_indices = self._split_to_micro_batch(
            batch,
            self.enable_dynamic_batch_size,
            max_tokens_per_mbs=self.max_tokens_per_mbs,
            split_num=num_sequences
            // self.cfg.algorithm.logprob_forward_micro_batch_size,
        )
        if self.enable_dynamic_batch_size:
            indices = sum(dbs_indices, [])
            revert_indices = torch.tensor(
                get_reverse_idx(indices),
                dtype=torch.long,
            )
        micro_batches = list(micro_batches_iter)

        recomputed_logprobs, ref_logprobs = None, None

        # Recompute logprobs
        recomputed_logprobs = torch.cat(
            [self.forward_batch(batch) for batch in micro_batches]
        ).cpu()

        if self.enable_dynamic_batch_size:
            assert len(indices) == recomputed_logprobs.size(0), (
                f"Dynamic batch size indices length {len(indices)} does not equal "
                f"output length {recomputed_logprobs.size(0)}"
            )
            recomputed_logprobs = recomputed_logprobs[revert_indices]

        # Ref logprobs
        if compute_ref_logprobs:
            assert self.ref_policy_state_dict is not None, (
                "Reference policy state dict is None but compute_ref_logprobs is True"
            )
            with cpu_weight_swap(
                self.model,
                self.ref_policy_state_dict,
                self.offload_model_buffer,
            ):
                ref_logprobs = torch.cat(
                    [self.forward_batch(batch) for batch in micro_batches]
                ).cpu()

                if self.enable_dynamic_batch_size:
                    assert len(indices) == ref_logprobs.size(0), (
                        f"Dynamic batch size indices length {len(indices)} does not equal "
                        f"output length {ref_logprobs.size(0)}"
                    )
                    ref_logprobs = ref_logprobs[revert_indices]

        return recomputed_logprobs, ref_logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
        do_offload=False,
    ):
        """
        Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
            do_offload: Whether offload weights after inference is done
        """
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )

        inference_split = self.cfg.actor.get("inference_split", None)
        if inference_split is None:
            if not self.is_pipeline:
                inference_split = 1
            else:
                inference_split = self.cfg.algorithm.n_minibatches
        assert self.total_batch_size_per_dp % inference_split == 0, (
            f"FSDPActor: total_batch_size_per_dp[{self.total_batch_size_per_dp}] should be divisible by inference_split[{inference_split}]"
        )

        min_result_len = 1
        max_result_len = (
            self.cfg.data.rollout_batch_size // self._world_size // inference_split
        )
        if not self.is_pipeline:
            min_result_len = max_result_len
            coll_rollout_results = []
        total_result_len = 0
        total_result_len_per_dp = self.cfg.data.rollout_batch_size // self._world_size
        cliped_results, unfinished_result = [], None
        while total_result_len < total_result_len_per_dp:
            batch, rollout_result, result_len, cliped_results, unfinished_result = (
                self.get_dynamic_batch_as_much(
                    input_channel,
                    min(min_result_len, total_result_len_per_dp - total_result_len),
                    min(max_result_len, total_result_len_per_dp - total_result_len),
                    cliped_results,
                    unfinished_result,
                )
            )
            total_result_len += result_len
            self.log_debug(
                f"[dynamic inference rank-{self._rank}] inference result_len={result_len}, total_result_len={total_result_len}/{total_result_len_per_dp}"
            )
            self._load_weight_and_optimizer()
            self.model.eval()

            with self.worker_timer():
                with torch.no_grad():
                    recomputed_logprobs, ref_logprobs = self.inference_step(
                        batch, rollout_result.num_sequence, compute_ref_logprobs
                    )

                rollout_result.recomputed_logprobs = recomputed_logprobs

                # Ref logprobs
                if compute_ref_logprobs:
                    rollout_result.ref_logprobs = ref_logprobs

            if self.is_pipeline:
                # for pipeline mode, send after inference to reduce latency.
                # should do split to ensure actor won't get too much batches.
                split_results = RolloutResult.split_results(rollout_result, result_len)
                for split_result in split_results:
                    output_channel.put(split_result, async_op=True)
            else:
                coll_rollout_results.append(rollout_result)

        if not self.is_pipeline:
            # for coll mode, merge results to reduce send time.
            rollout_result = RolloutResult.merge_result_list(coll_rollout_results)
            split_results = RolloutResult.split_results(
                rollout_result,
                min(total_result_len, self.cfg.algorithm.n_minibatches),
            )
            for split_result in split_results:
                output_channel.put(split_result)
        assert total_result_len == total_result_len_per_dp, (
            f"Expected {total_result_len_per_dp} sequences from channel, but got {total_result_len}"
        )

    @Worker.timer("training_step")
    def training_step(
        self, batch: dict[str, torch.Tensor] | BatchResizingIterator
    ) -> tuple[dict[str, torch.Tensor], float, list[float]]:
        if isinstance(batch, dict):
            global_batch_size = batch["input_ids"].shape[0]
            assert global_batch_size % self.micro_batch_size == 0, (
                f"global batch size {global_batch_size} can not divide micro_batch_size {self.micro_batch_size}"
            )
            micro_batches_iter, micro_batch_cnt, _ = self._split_to_micro_batch(
                batch,
                self.enable_dynamic_batch_size,
                max_tokens_per_mbs=self.max_tokens_per_mbs,
                split_num=global_batch_size // self.micro_batch_size,
            )
            self.gradient_accumulation = micro_batch_cnt
        else:
            global_batch_size = self.total_batch_size_per_dp // self.n_mini_batches
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt

            def iterator_wrapper():
                for _ in range(micro_batch_cnt):
                    yield next(batch)

            micro_batches_iter = iterator_wrapper()
        self.optimizer.zero_grad()
        mbs_metrics_list = {}
        for idx, m_batch in enumerate(micro_batches_iter):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == micro_batch_cnt,
            )
            for k, v in m_batch.items():
                m_batch[k] = (
                    v.to(Worker.torch_device_type) if isinstance(v, torch.Tensor) else v
                )

            # batch for forward
            logprobs, entropy = self.forward_batch(m_batch, True)

            # batch for backward
            # Prefer recomputed_logprobs (from actor inference), fallback to rollout_logprobs
            old_logprobs = m_batch.get("recomputed_logprobs")
            if old_logprobs is None:
                old_logprobs = m_batch["rollout_logprobs"]
            advantages = m_batch["advantages"]
            ref_logprobs = None
            if "ref_logprobs" in m_batch:
                ref_logprobs = m_batch["ref_logprobs"]

            loss_mask = m_batch["response_mask"][:, -self.response_len :]

            clip_ratio = self.cfg.algorithm.ratio_clip_eps
            clip_ratio_low = self.cfg.algorithm.get("clip_ratio_low", None)
            clip_ratio_high = self.cfg.algorithm.get("clip_ratio_high", None)
            clip_ratio_low = (
                clip_ratio_low if clip_ratio_low is not None else clip_ratio
            )
            clip_ratio_high = (
                clip_ratio_high if clip_ratio_high is not None else clip_ratio
            )
            clip_ratio_c = self.cfg.algorithm.get("clip_ratio_c", 3.0)

            if self.cfg.algorithm.get("importance_sampling_fix", False):
                if (
                    "rollout_logprobs" not in m_batch
                    or "recomputed_logprobs" not in m_batch
                ):
                    raise ValueError(
                        "importance_sampling_fix requires both rollout_logprobs and recomputed_logprobs"
                    )
                rollout_logprobs = m_batch["rollout_logprobs"]
                recomputed_logprobs = m_batch["recomputed_logprobs"]
                advantages = advantages * torch.clamp(
                    (recomputed_logprobs - rollout_logprobs).exp(),
                    max=self.cfg.algorithm.importance_sampling_clip,
                )

            loss, mbs_metrics_data = policy_loss(
                task_type=self.task_type,
                loss_type=self.cfg.algorithm.loss_type,
                loss_agg_func=self.loss_agg_func,
                logprobs=logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                clip_ratio_c=clip_ratio_c,
                clip_ratio_low=clip_ratio_low,
                clip_ratio_high=clip_ratio_high,
                loss_mask=loss_mask,
                clip_log_ratio_min=self.cfg.algorithm.get("clip_log_ratio_min", None),
                clip_log_ratio_max=self.cfg.algorithm.get("clip_log_ratio_max", None),
                fast_path_zero_loss_mask=True,
            )

            entropy_loss = torch.tensor(
                0.0, device=Worker.torch_platform.current_device()
            )
            if self.calculate_entropy:
                entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
                if self.calculate_entropy_loss:
                    loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

            kl_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
            if self.kl_beta > 0 and ref_logprobs is not None:
                kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
                kl_loss = self.loss_agg_func(kld, loss_mask)
                loss = loss + kl_loss * self.kl_beta

            # add to log
            # scale loss for gradient accumulation and backprop
            final_loss_metric = loss.detach()
            loss = loss / self.gradient_accumulation
            with backward_ctx:
                self.grad_scaler.scale(loss).backward()

            mbs_metrics_data.update(
                {
                    "actor/final_loss": final_loss_metric,
                    "actor/entropy_loss": entropy_loss.detach(),
                    "actor/kl_loss": kl_loss.detach(),
                }
            )

            append_to_dict(mbs_metrics_list, mbs_metrics_data)

        grad_norm, lr_list = self.optimizer_step()

        if self.lr_sched_sync_with_optim:
            self.lr_scheduler.step()

        # display the degree of mismatch between training and rollout
        rollout_train_kl = compute_rollout_train_kl(m_batch, loss_mask)

        # aggregate metrics across micro-batches
        explained_variance_stats = pop_critic_explained_variance_stats(mbs_metrics_list)
        mean_metric_dict = {
            key: torch.mean(torch.stack(value))
            for key, value in mbs_metrics_list.items()
        }
        if rollout_train_kl is not None:
            mean_metric_dict["actor/rollout_train_kl"] = rollout_train_kl

        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        if explained_variance_stats:
            reduced_stats = all_reduce_dict(
                explained_variance_stats, op=torch.distributed.ReduceOp.SUM
            )
            mean_metric_dict[CRITIC_EXPLAINED_VARIANCE_KEY] = (
                compute_critic_explained_variance_from_stats(reduced_stats).item()
            )

        mean_metric_dict["actor/grad_norm"] = float(grad_norm)
        mean_metric_dict["actor/lr"] = lr_list[0]
        return mean_metric_dict

    def run_training_pipeline(self, input_channel: Channel) -> tuple[dict, list]:
        self.model.train()
        train_batch_iterator = BatchResizingIterator(
            cfg=self.cfg,
            get_batch_fn=partial(self.get_batch, input_channel),
            micro_batch_size=self.micro_batch_size,
            total_batch_size=self.total_batch_size_per_dp,
            num_global_batches=self.n_mini_batches,
            forward_only=False,
        )
        train_batch_iterator.register_get_batch_handler(
            self.compute_advantages_and_returns
        )

        if self.cfg.algorithm.normalize_advantages:

            def normalize_advantages(batch: dict[str, torch.Tensor]):
                mask = batch["response_mask"][:, -self.response_len :]
                batch["advantages"] = masked_normalization(batch["advantages"], mask)
                return batch

            train_batch_iterator.register_global_batch_handler(normalize_advantages)

        self._load_weight_and_optimizer()
        training_metrics_list = []
        with self.worker_timer("run_training"):
            for _ in range(self.n_mini_batches):
                mean_metric_dict = self.training_step(batch=train_batch_iterator)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        batch = train_batch_iterator.get_all_batches()
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    def _dp_load_balance(self, batch: dict[str, torch.Tensor]):
        batch_size = batch["input_ids"].shape[0]
        assert batch_size == self.total_batch_size_per_dp, (
            f"DP Load balance is only available when a single batch contains all data, e.g., in collocated mode. But got {batch_size=} and {self.total_batch_size_per_dp=}."
        )
        batch = RolloutDataBalance.from_rollout_batches(
            rollout_batches=batch,
            dp_world_size=torch.distributed.get_world_size(),
            dp_rank=torch.distributed.get_rank(),
            dp_group=torch.distributed.group.WORLD,
            partitioning_tool=get_seqlen_balanced_partitions,
        )
        return batch

    @Worker.timer("run_training")
    def run_training(
        self, input_channel: Channel, do_offload=False
    ) -> tuple[dict, list]:
        # Get all batches for this DP
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )

        if self.is_pipeline:
            return self.run_training_pipeline(input_channel)

        batches = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        global_batch = RolloutResult.merge_batches(batches)

        assert (
            "recomputed_logprobs" in global_batch or "rollout_logprobs" in global_batch
        )

        # Compute advantages and returns
        global_batch = self.compute_advantages_and_returns(global_batch)

        if self.enable_dp_load_balance:
            global_batch = self._dp_load_balance(global_batch)

        if self.cfg.algorithm.normalize_advantages:
            mask = global_batch["response_mask"][:, -self.response_len :]
            global_batch["advantages"] = masked_normalization(
                global_batch["advantages"], mask
            )

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()

        mini_batches = get_iterator_k_split(
            global_batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        training_metrics_list = []
        # Global batch iterations
        with self.worker_timer():
            for mini_batch in mini_batches:
                mean_metric_dict = self.training_step(batch=mini_batch)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            global_batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    @Worker.timer("compute_advantages_and_returns")
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["response_mask"][:, -self.response_len :]
                logprob = batch.get("recomputed_logprobs")
                if logprob is None:
                    logprob = batch.get("rollout_logprobs")
                logprob = logprob.to(Worker.torch_device_type)

                advantages, _ = calculate_adv_and_returns(
                    task_type=self.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    rewards=batch["rewards"].to(Worker.torch_device_type),
                    loss_mask=mask.to(Worker.torch_device_type),
                    group_size=self.cfg.algorithm.group_size,
                    kl_beta=self.reinpp_kl_beta,
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=logprob,
                    ref_logprob=batch["ref_logprobs"].to(Worker.torch_device_type)
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = advantages
        return batch


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self._opd_teacher_model = None
        self._routed_teacher_models: dict[str, Any] = {}
        self._opd_student_metadata = None
        self._opd_coordinate_log_keys: set[str] = set()
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")

        self.enable_sft_co_train = cfg.actor.get("enable_sft_co_train", False)
        self.version = 0
        if self.enable_sft_co_train:
            self._build_sft_data_loader()

        # create weight syncer
        weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer")
        self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )
        self.update_epoch = self.cfg.algorithm.get("update_epoch", 1)

        self._sync_weight_comm_options = self.weight_syncer.comm_options

        self._is_weight_sender = self._rank == 0
        self._actor_world_size = self._world_size
        self._rollout_all_ranks = list(
            range(self._component_placement.get_world_size("rollout"))
        )

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        self.setup_model_and_optimizer()

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        return model

    def get_rollout_state_dict(self) -> dict:
        return self.get_model_state_dict(cpu_offload=False, full_state_dict=False)

    @Worker.timer("actor/sync_model_to_rollout")
    async def sync_model_to_rollout(self) -> None:
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        state_dict = self.get_rollout_state_dict()

        async def send_func(data):
            if not self._is_weight_sender:
                return
            await self.broadcast(
                data,
                groups=[
                    (self._group_name, 0),
                    (self._rollout_group_name, self._rollout_all_ranks),
                ],
                src=(self._group_name, 0),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        async def recv_func():
            return await self.recv(
                src_group_name=self._rollout_group_name,
                src_rank=0,
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        if not self.weight_syncer.sender_initialized():
            await self.weight_syncer.init_sender(
                state_dict=state_dict,
                send=send_func,
                recv=recv_func,
                param_names_need_sync=self.param_names_need_sync,
                is_sender=self._is_weight_sender,
            )

        version = (
            self.get_rollout_sync_version()
            if hasattr(self, "get_rollout_sync_version")
            else self.version
        )
        await self.weight_syncer.sync(state_dict, send_func, version=version)

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad(True)

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.rollout_batch = convert_trajectories_to_batch(recv_list)

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.env.train.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    @Worker.timer("actor/compute_adv")
    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """
        if self.cfg.algorithm.loss_type in VELOCITY_DISTILL_LOSS_TYPES:
            # Velocity distillation (Method 1) and analytic local KL (Method 3 /
            # M3-Hybrid) are supervised: no reward / advantage / GAE. Only the frozen
            # teacher's velocities are needed, computed once post-rollout.
            self.compute_teacher_velocities()
            return {}

        if self.cfg.algorithm.adv_type == "opd":
            if SupportedModel(self.cfg.actor.model.model_type) in FLOW_MODELS:
                # Method 4: the teacher scores the student's sampled flow transition. The
                # OpenVLA path below scores action *tokens* and does not apply here.
                self.compute_teacher_velocities()
            else:
                self.compute_opd_teacher_logprobs()

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            # For flow-model OPD the advantage compares against the sampled step only;
            # see _opd_prev_logprobs_at_sampled_step for why this is a separate key.
            "prev_logprobs": self.rollout_batch.get(
                "opd_prev_logprobs", self.rollout_batch.get("prev_logprobs", None)
            ),
            "teacher_logprobs": self.rollout_batch.get("teacher_logprobs", None),
            "num_action_chunks": self.cfg.actor.model.num_action_chunks,
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
        }

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    @Worker.timer("actor/compute_opd_teacher_logprobs")
    def compute_opd_teacher_logprobs(self) -> None:
        assert self.rollout_batch.get("teacher_logprobs", None) is None, (
            "OPD teacher_logprobs must be computed after rollout on actor workers."
        )
        assert self.cfg.rollout.get("expert_model", None) is not None, (
            "OPD requires rollout.expert_model as teacher model config."
        )
        assert "forward_inputs" in self.rollout_batch, (
            "OPD teacher logprob computation requires rollout forward_inputs."
        )
        assert "prev_logprobs" in self.rollout_batch, (
            "OPD teacher logprob computation requires student prev_logprobs."
        )
        assert SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ], "OPD teacher logprob computation currently supports OpenVLA models."

        prev_logprobs = self.rollout_batch["prev_logprobs"]
        time_dim, batch_dim = prev_logprobs.shape[:2]
        flat_batch_size = time_dim * batch_dim

        assert self.enable_offload and self.is_weight_offloaded, (
            "OPD teacher logprob computation expects actor weights to be "
            "offloaded before moving the teacher model to GPU."
        )
        teacher_model = self._get_opd_teacher_model()
        teacher_model.to(self.device)

        flat_forward_inputs = flatten_nested_tensor_time_batch(
            self.rollout_batch["forward_inputs"], ("forward_inputs",)
        )
        num_chunks = (
            flat_batch_size + self.cfg.actor.micro_batch_size - 1
        ) // self.cfg.actor.micro_batch_size
        teacher_logprobs = []
        kwargs = {
            "temperature": self.cfg.rollout.sampling_params.temperature_train,
            "top_k": self.cfg.rollout.sampling_params.top_k,
        }
        with torch.no_grad():
            for micro_batch in split_dict_to_chunk(flat_forward_inputs, num_chunks):
                micro_batch = put_tensor_device(micro_batch, self.device)
                with self.amp_context:
                    teacher_output = teacher_model(
                        forward_inputs=micro_batch,
                        compute_logprobs=True,
                        compute_entropy=False,
                        compute_values=False,
                        use_cache=False,
                        **kwargs,
                    )
                teacher_logprobs.append(teacher_output["logprobs"].detach().cpu())

        teacher_logprobs = torch.cat(teacher_logprobs, dim=0)
        expected_shape = (flat_batch_size, *prev_logprobs.shape[2:])
        assert teacher_logprobs.shape == expected_shape, (
            f"teacher_logprobs shape {teacher_logprobs.shape} must match "
            f"flattened student logprobs shape {expected_shape}."
        )
        self.rollout_batch["teacher_logprobs"] = teacher_logprobs.reshape(
            time_dim, batch_dim, *teacher_logprobs.shape[1:]
        )

        teacher_model.to("cpu")
        clear_memory()

    def _get_opd_teacher_model(self):
        if self._opd_teacher_model is None:
            teacher_model_config = build_expert_model_config(
                self.cfg, self.cfg.actor.model
            )
            teacher_model = get_model(teacher_model_config)
            if self.cfg.runner.get("expert_ckpt_path", None):
                teacher_model_dict = torch.load(
                    self.cfg.runner.expert_ckpt_path, map_location="cpu"
                )
                teacher_model.load_state_dict(teacher_model_dict)
            teacher_model.eval()
            teacher_model.requires_grad_(False)
            teacher_model.to("cpu")
            self._opd_teacher_model = teacher_model
        return self._opd_teacher_model

    def _velocity_distill_flow_spec(self):
        """Flow conventions the velocity-distillation code needs, per model family.

        Returns ``(num_steps, joint_logprob, timestep_values)`` where ``timestep_values[k]``
        is what the model's NFT forward expects as the timestep for ``chains[k]``.
        OpenPI runs continuous tau from 1 (noise) down to 1/num_steps; GR00T takes the
        integer denoise index and converts internally, with time running the other way
        (0 = noise, 1 = data). Both index ``chains`` identically, so only this mapping and
        the config key names differ.
        """
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type in [SupportedModel.OPENPI]:
            num_steps = int(self.cfg.actor.model.num_steps)
            joint_logprob = bool(self.cfg.actor.model.openpi.joint_logprob)
            timestep_values = torch.linspace(1.0, 1.0 / num_steps, num_steps)
        elif model_type in [SupportedModel.GR00T]:
            num_steps = int(self.cfg.actor.model.denoising_steps)
            joint_logprob = bool(self.cfg.actor.model.rl_head_config.joint_logprob)
            timestep_values = torch.arange(num_steps, dtype=torch.float32)
        else:
            raise NotImplementedError(
                f"velocity distillation does not support {model_type} yet."
            )
        return num_steps, joint_logprob, timestep_values

    def _flow_sde_transition_spec(self):
        """Per-model-family inputs to the shared ``flow_sde`` transition coefficients.

        ``flow_sde_transition_coeffs`` was written against OpenPI, but GR00T's ``flow_sde``
        branch was checked against it numerically and agrees on ``w0 + w1``, ``sigma`` and
        ``|c|``, indexed by the same denoise step. Only ``c``'s sign differs (GR00T +,
        OpenPI -), which cancels in M3's ``w = c**2 / (2 sigma**2)`` but not in M4's
        ``mu_T``. Both reduce to ``c = delta * (1 + noise_level**2 / 2)``.

        Returns:
            ``(noise_level, action_chunk, action_env_dim, c_sign)``.
        """
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type in [SupportedModel.OPENPI]:
            openpi_cfg = self.cfg.actor.model.openpi
            return (
                float(openpi_cfg.noise_level),
                int(openpi_cfg.action_chunk),
                int(openpi_cfg.action_env_dim),
                1.0,
            )
        elif model_type in [SupportedModel.GR00T]:
            return (
                float(self.cfg.actor.model.rl_head_config.noise_level),
                int(self.cfg.actor.model.num_action_chunks),
                int(self.cfg.actor.model.action_dim),
                -1.0,
            )
        else:
            raise NotImplementedError(
                f"flow_sde transition coefficients are not defined for {model_type}."
            )

    def _teacher_names(self) -> list[str]:
        """Configured teachers, in the order the rollout worker indexed them."""
        expert_models = self.cfg.rollout.get("expert_models", None)
        return list(expert_models.keys()) if expert_models else []

    def _get_routed_teacher_model(self, name: str):
        """Load (and cache on CPU) the teacher a task group routes to.

        Teachers stay resident in host memory and are moved to the GPU one at a time --
        reloading ~8GB of weights from the shared filesystem on every training step would
        dominate the step time, while three cached teachers cost ~23GB of host RAM per rank.
        """
        if name not in self._routed_teacher_models:
            overrides = self.cfg.rollout.expert_models[name]
            teacher_model_config = copy.deepcopy(self.cfg.actor.model)
            with open_dict(teacher_model_config):
                for key, value in overrides.items():
                    teacher_model_config[key] = value
            teacher_model = get_model(teacher_model_config)
            teacher_model.eval()
            teacher_model.requires_grad_(False)
            teacher_model.to("cpu")
            self._routed_teacher_models[name] = teacher_model
            self.log_info(f"Loaded routed OPD teacher {name!r}.")
        return self._routed_teacher_models[name]

    def _get_opd_student_gr00t_metadata(self):
        """Load the student's normalization metadata once on each actor rank."""
        if self._opd_student_metadata is None:
            from gr00t.data.schema import DatasetMetadata

            path = (
                Path(self.cfg.actor.model.model_path)
                / "experiment_cfg"
                / "metadata.json"
            )
            with open(path) as metadata_file:
                all_metadata = json.load(metadata_file)
            tag = str(self.cfg.actor.model.embodiment_tag)
            assert tag in all_metadata, f"No {tag!r} metadata in {path}."
            self._opd_student_metadata = DatasetMetadata.model_validate(
                all_metadata[tag]
            )
        return self._opd_student_metadata

    def _suite_balanced_sample(
        self, flat_batch_size: int, time_dim: int, batch_dim: int
    ) -> Optional[torch.Tensor]:
        """Draw a suite-balanced subset of the flat rollout, or None to use all of it.

        Enabled by ``algorithm.suite_balanced_sampling``. Every task suite then contributes
        the same number of samples to the update, so each routed teacher gets the same share
        of the gradient regardless of how many transitions its tasks happened to produce.
        Without it the share is set by episode length: RoboCasa's composite horizons are
        ~3x the atomic ones, so the atomic teacher would carry ~1/8 of the weight rather
        than 1/3.

        Drawn without replacement where a suite's pool covers its quota, and with
        replacement only for the shortfall. A shortfall on the short-horizon suite is the
        normal case, not an edge case -- it has the smallest pool by construction.
        Duplicates raise variance but not bias, and the pool is regenerated from fresh
        rollouts every step.

        Returns flat indices into the ``[T, B]``-major ordering that both
        ``flatten_nested_tensor_time_batch`` and ``process_nested_dict_for_train`` use.
        """
        if not self.cfg.algorithm.get("suite_balanced_sampling", False):
            return None

        forward_inputs = self.rollout_batch["forward_inputs"]
        teacher_ids = forward_inputs.get("teacher_ids", None)
        assert teacher_ids is not None, (
            "suite_balanced_sampling needs per-sample teacher_ids, which the rollout "
            "worker attaches when rollout.expert_models is set."
        )
        teacher_ids = teacher_ids.reshape(-1).cpu().long()

        # Steps past `done` carry no valid transition. loss_mask marks them; it exists
        # whenever auto_reset and ignore_terminations are both False, which is what the
        # velocity-distillation configs use.
        loss_mask = self.rollout_batch.get("loss_mask", None)
        valid = build_flat_valid_transition_mask(
            flat_batch_size, time_dim, batch_dim, loss_mask
        )

        target = int(self.cfg.actor.global_batch_size) // self._world_size
        # A per-rank quota is globally balanced only if every participating rank holds
        # every suite. Deriving this list from local ids made a missing suite invisible:
        # that suite was absent from both the loop and the assertion. Gather the union of
        # suites produced across ranks, then require each rank to cover that union.
        names = self._teacher_names()
        expected = teacher_suite_presence(teacher_ids, len(names))
        if torch.distributed.is_initialized():
            device = Worker.torch_platform.current_device()
            flags = expected.to(device)
            torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX)
            expected = flags.cpu()
        suites = torch.nonzero(expected, as_tuple=False).reshape(-1).tolist()

        present = set(teacher_ids.tolist())
        missing = [names[i] for i in suites if i not in present]
        assert not missing, (
            f"rank {self._rank} holds no samples for suite(s) {missing}, which other "
            "ranks do produce. The quota is drawn per rank, so an unbalanced split makes "
            "the global weighting the mean of the per-rank ones while every local log "
            "line still reads as balanced. Raise total_num_envs, change task placement, "
            "use env.train.task_sampling_strategy=suite_balanced for RoboCasa365, "
            "or turn algorithm.suite_balanced_sampling off and accept natural horizon "
            "weighting."
        )
        # A fresh draw every step. Seeding on rank alone would redraw the same indices
        # forever, which no metric would reveal.
        self._sample_draw_count = getattr(self, "_sample_draw_count", 0) + 1
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank + self._sample_draw_count * 7919)

        chosen, drawn, distinct = [], {}, {}
        for i, suite in enumerate(suites):
            pool = torch.nonzero(
                (teacher_ids == suite) & valid, as_tuple=False
            ).reshape(-1)
            # Spread the remainder over the first suites rather than dropping it.
            quota = target // len(suites) + (1 if i < target % len(suites) else 0)
            assert pool.numel() > 0, (
                f"suite {self._teacher_names()[suite]!r} produced no valid transitions on "
                f"rank {self._rank}. The quota is drawn per rank, so every rank's envs must "
                "cover every suite. Use RoboCasa365 suite_balanced task sampling with "
                "groups per worker divisible by the number of suites."
            )
            if pool.numel() >= quota:
                take = pool[torch.randperm(pool.numel(), generator=g)[:quota]]
            else:
                extra = torch.randint(
                    0, pool.numel(), (quota - pool.numel(),), generator=g
                )
                take = torch.cat([pool, pool[extra]])
            chosen.append(take)
            drawn[self._teacher_names()[suite]] = int(take.numel())
            distinct[self._teacher_names()[suite]] = int(min(pool.numel(), quota))

        idx = torch.cat(chosen)
        # Shuffle so a micro-batch is not a single suite.
        idx = idx[torch.randperm(idx.numel(), generator=g)]
        self.log_info(
            f"Suite-balanced sample: drawn {drawn}, distinct {distinct}, "
            f"total {idx.numel()} of {int(valid.sum())} valid."
        )
        return idx

    @Worker.timer("actor/compute_teacher_velocities")
    def compute_teacher_velocities(self, already_sampled: bool = False) -> None:
        """Method 1 (velocity distillation): compute the frozen teacher velocity
        ``v_T(A_t^tau, o_t, tau)`` at student-visited flow states, once after rollout.

        One uniformly-random denoise step ``tau`` is selected per sample (uniform timestep
        weighting). ``x_t`` is taken from the student's on-policy ODE trajectory ``chains``.
        The same ``(x_t, tau)`` are cached so the student recomputes ``v_S`` at the identical
        point during training. Teacher is offloaded afterwards (mirrors OPD).
        """
        assert self.rollout_batch.get("teacher_velocities", None) is None, (
            "teacher_velocities must be computed once after rollout."
        )
        assert self.cfg.rollout.get(
            "expert_model", None
        ) is not None or self.cfg.rollout.get("expert_models", None), (
            "velocity distillation requires a teacher: rollout.expert_model for a single "
            "teacher, or rollout.expert_models for routed multi-teacher OPD."
        )
        assert "forward_inputs" in self.rollout_batch, (
            "velocity distillation requires rollout forward_inputs (chains + obs)."
        )
        assert SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.GR00T,
        ], "velocity distillation currently supports OpenPI and GR00T N1.5 models."

        num_steps, joint_logprob, timestep_values = self._velocity_distill_flow_spec()
        forward_inputs = self.rollout_batch["forward_inputs"]
        # chains stores the initial noise plus one state per denoise step, so its step
        # axis is num_steps + 1 (openpi_action_model._sample_actions_with_prefix_cache;
        # gr00t_action_model.get_rl_action seeds `chains` with the initial noise too).
        chains = forward_inputs["chains"]  # [T, B, num_steps + 1, horizon, dim]
        time_dim, batch_dim = chains.shape[0], chains.shape[1]
        assert chains.shape[2] == num_steps + 1, (
            f"chains step axis {chains.shape[2]} must be num_steps + 1 "
            f"({num_steps} + 1); the flow timestep convention depends on it."
        )
        flat_batch_size = time_dim * batch_dim

        flat_forward_inputs = flatten_nested_tensor_time_batch(
            forward_inputs, ("forward_inputs",)
        )
        flat_chains = flat_forward_inputs[
            "chains"
        ]  # [flat, num_steps + 1, horizon, dim]

        # Which denoise step to supervise. Index k in [0, num_steps) pairs chains[k] with
        # timesteps[k], matching the model's own NFT convention
        # (openpi_action_model._init_nft_state / sample_mean_var_val): chains[k] is the
        # state the model evaluates at t_input = timesteps[k]. The final entry
        # chains[num_steps] is the emitted action x_0 (t = 0), which the rollout never
        # evaluates a velocity at, so it is excluded.
        #
        # A Hybrid-SDE rollout (joint_logprob=False) injects noise at exactly one sampled
        # step k, and that transition -- condition on x_k at tau_k, sample x_{k+1} ~
        # N(mu_S, Sigma_k) -- is the one M4's sampled reverse-KL is defined on, and the one
        # the rollout's prev_logprobs scores. Supervising there makes M3-Hybrid M4's
        # control, differing only in analytic-vs-sampled. A pure ODE rollout has no
        # distinguished step, so draw one uniformly per sample instead.
        if joint_logprob:
            self._timestep_draw_count = getattr(self, "_timestep_draw_count", 0) + 1
            step_idx = sample_flow_timestep_indices(
                flat_batch_size,
                num_steps,
                seed=self.cfg.actor.seed,
                rank=self._rank,
                draw_count=self._timestep_draw_count,
            )
        else:
            step_idx = flat_forward_inputs["denoise_inds"][:, 0].cpu().long()
            assert step_idx.shape[0] == flat_batch_size, (
                f"denoise_inds gave {step_idx.shape[0]} steps for "
                f"{flat_batch_size} samples."
            )
            assert int(step_idx.min()) >= 0 and int(step_idx.max()) < num_steps, (
                f"denoise_inds out of range [0, {num_steps}): "
                f"[{int(step_idx.min())}, {int(step_idx.max())}]. A rollout in eval mode "
                "stores -1 and must not be used for teacher supervision."
            )
        arange = torch.arange(flat_batch_size)
        x_t = flat_chains[arange, step_idx].contiguous()  # [flat, horizon, dim]
        # What the model's NFT forward wants as "the timestep" for chains[k]; see
        # _velocity_distill_flow_spec.
        tau = timestep_values[step_idx].contiguous()  # [flat]

        # Carry x_t / tau through the same micro-batch split as the obs.
        flat_forward_inputs["_nft_x_t"] = x_t
        flat_forward_inputs["_nft_tau"] = tau

        # Suite-balanced sampling, when enabled, decides here which transitions the update
        # will use. Scoring only those caps the teacher pass, and run_training reuses the
        # same indices so the student is not run on transitions the loss will ignore.
        # Held on the worker, not in rollout_batch: that dict is walked by several
        # helpers that assume every entry is a [T, B, ...] tensor, and a 1-D index would
        # be reshaped along with them.
        sample_idx = (
            torch.arange(flat_batch_size)
            if already_sampled
            else self._suite_balanced_sample(flat_batch_size, time_dim, batch_dim)
        )
        self._sample_idx = sample_idx

        teacher_ids = flat_forward_inputs.get("teacher_ids", None)
        teacher_names = self._teacher_names()
        timestep_counts = count_flow_timesteps_by_suite(
            step_idx,
            num_steps,
            teacher_ids=teacher_ids,
            sample_idx=sample_idx,
            num_teachers=len(teacher_names) if teacher_ids is not None else None,
        )
        if torch.distributed.is_initialized():
            device = Worker.torch_platform.current_device()
            device_counts = timestep_counts.to(device)
            torch.distributed.all_reduce(
                device_counts, op=torch.distributed.ReduceOp.SUM
            )
            timestep_counts = device_counts.cpu()
        labels = teacher_names if teacher_ids is not None else ["all"]
        assert len(labels) == timestep_counts.shape[0], (
            f"Have {len(labels)} suite labels for {timestep_counts.shape[0]} "
            "timestep-count rows."
        )
        count_summary = {
            label: counts.tolist()
            for label, counts in zip(labels, timestep_counts, strict=True)
        }
        self.log_info(
            f"Flow supervision counts by suite and denoise step: {count_summary}."
        )

        if sample_idx is None:
            teacher_velocities = self._teacher_velocity_pass(
                flat_forward_inputs, flat_batch_size
            )  # [flat, chunk, dim]
        else:
            subset = {
                key: value[sample_idx] if torch.is_tensor(value) else value
                for key, value in flat_forward_inputs.items()
            }
            subset_velocities = self._teacher_velocity_pass(
                subset, int(sample_idx.numel())
            )
            # Scatter back into the full flat layout so the [T, B] reshape below and the
            # indexing in run_training keep meaning the same thing. Rows that were not
            # drawn stay zero and are never read.
            teacher_velocities = torch.zeros(
                (flat_batch_size, *subset_velocities.shape[1:]),
                dtype=subset_velocities.dtype,
            )
            teacher_velocities[sample_idx] = subset_velocities

        self.rollout_batch["teacher_velocities"] = teacher_velocities.reshape(
            time_dim, batch_dim, *teacher_velocities.shape[1:]
        )
        self.rollout_batch["nft_x_t"] = x_t.reshape(time_dim, batch_dim, *x_t.shape[1:])
        self.rollout_batch["nft_timesteps"] = tau.reshape(time_dim, batch_dim)

        if self.cfg.algorithm.loss_type == "analytic_local_kl":
            # w(tau) for the local Flow-SDE analytic KL (M3 / M3-Hybrid). Built from the
            # closed-form flow_sde schedule even when the rollout is deterministic ODE --
            # see flow_sde_kl_weights for why.
            #
            # Formerly OPENPI-only: the two families' flow_sde transitions agree on
            # w = c**2 / (2 sigma**2). See _flow_sde_transition_spec.
            noise_level, _, _, _ = self._flow_sde_transition_spec()
            weights = flow_sde_kl_weights(
                num_steps=num_steps,
                noise_level=noise_level,
            )
            kl_weight = weights[step_idx].to(torch.float32)
            self.rollout_batch["nft_kl_weight"] = kl_weight.reshape(time_dim, batch_dim)

        if self.cfg.algorithm.adv_type == "opd":
            # Formerly OPENPI-only; unlike M3 this needs the signed c, see
            # _teacher_transition_logprobs.
            # M4: the teacher's log-probability of the transition the student actually
            # sampled, l_T = log N(x_{k+1}; mu_T, Sigma_k). Reuses the teacher velocities
            # computed above -- mu_T only differs from mu_S through v_T -- so M4 costs the
            # same single teacher forward as M1/M3. Sigma_k comes from the student's
            # schedule so teacher and student share it, as the reverse KL requires.
            teacher_logprobs = self._teacher_transition_logprobs(
                flat_chains=flat_chains,
                x_cur=x_t,
                step_idx=step_idx,
                teacher_velocities=teacher_velocities,
                num_steps=num_steps,
                time_dim=time_dim,
                batch_dim=batch_dim,
            )
            self.rollout_batch["teacher_logprobs"] = teacher_logprobs
            reduced = self._opd_prev_logprobs_at_sampled_step(step_idx)
            if reduced is not None:
                self.rollout_batch["opd_prev_logprobs"] = reduced
                self._log_opd_reward_per_suite(teacher_logprobs, reduced)

        clear_memory()

    def _teacher_velocity_pass(
        self,
        flat_forward_inputs: dict[str, torch.Tensor],
        flat_batch_size: int,
    ) -> torch.Tensor:
        """Score the frozen teacher(s) at the cached ``(x_t, tau)``.

        With a single ``rollout.expert_model`` this is one pass over the batch. With
        ``rollout.expert_models`` (routed multi-teacher OPD) the batch is partitioned by
        the per-sample ``teacher_ids`` the rollout worker attached, and each teacher is
        moved to the GPU, run over its own partition, and moved back -- so peak GPU memory
        matches the single-teacher case however many teachers are configured.
        """

        def _run(model, inputs: dict[str, torch.Tensor], size: int) -> torch.Tensor:
            # Chunk count must come from this call's own size, not from the whole batch:
            # the routed path calls _run once per suite on a partition, and
            # split_dict_to_chunk returns fewer chunks than asked for when the size does
            # not divide, which then indexes past the end. Only latent while every
            # partition happened to be a multiple of the chunk count.
            transport = None
            if SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.GR00T:
                assert "opd_raw_states" in inputs, (
                    "Cross-checkpoint GR00T OPD needs the raw RoboCasa state saved by "
                    "the rollout model."
                )
                teacher_inputs = dict(inputs)
                teacher_state, teacher_state_mask = model.opd_normalize_robocasa_state(
                    inputs["opd_raw_states"]
                )
                teacher_inputs["state"] = teacher_state.to(dtype=inputs["state"].dtype)
                teacher_inputs["state_mask"] = teacher_state_mask

                endpoint_scale, endpoint_shift = model.opd_action_endpoint_affine_to(
                    self._get_opd_student_gr00t_metadata()
                )
                differs = not (
                    torch.equal(endpoint_scale, torch.ones_like(endpoint_scale))
                    and torch.equal(endpoint_shift, torch.zeros_like(endpoint_shift))
                )
                if differs and (
                    self.cfg.algorithm.loss_type == "analytic_local_kl"
                    or self.cfg.algorithm.adv_type == "opd"
                ):
                    raise NotImplementedError(
                        "Cross-metadata GR00T M3/M4 needs the transported teacher "
                        "covariance/Jacobian, not only a transported velocity. Use M1 "
                        "until that KL is implemented."
                    )
                source_x_t = map_target_flow_state_to_source(
                    inputs["_nft_x_t"],
                    inputs["_nft_tau"],
                    int(self.cfg.actor.model.denoising_steps),
                    endpoint_scale,
                    endpoint_shift,
                )
                teacher_inputs["_nft_x_t"] = source_x_t
                inputs = teacher_inputs
                transport = (
                    source_x_t,
                    inputs["_nft_tau"],
                    endpoint_scale,
                    endpoint_shift,
                )
                log_key = str(model.model_path)
                if log_key not in self._opd_coordinate_log_keys:
                    self._opd_coordinate_log_keys.add(log_key)
                    valid_dim = int(model.valid_action_dim)
                    self.log_info(
                        "GR00T OPD coordinate adapter active for "
                        f"{log_key}: endpoint scale "
                        f"[{float(endpoint_scale[:valid_dim].min()):.4g}, "
                        f"{float(endpoint_scale[:valid_dim].max()):.4g}], "
                        f"max |shift| "
                        f"{float(endpoint_shift[:valid_dim].abs().max()):.4g}."
                    )

            num_chunks = max(
                1,
                (size + self.cfg.actor.micro_batch_size - 1)
                // self.cfg.actor.micro_batch_size,
            )
            velocities = []
            with torch.no_grad():
                for micro_batch in split_dict_to_chunk(inputs, num_chunks):
                    micro_batch = put_tensor_device(micro_batch, self.device)
                    nft_inputs = {
                        "x_t": micro_batch["_nft_x_t"],
                        "timesteps": micro_batch["_nft_tau"],
                    }
                    with self.amp_context:
                        out = model(
                            forward_type=ForwardType.NFT,
                            forward_inputs=micro_batch,
                            nft_inputs=nft_inputs,
                            compute_values=False,
                        )
                    velocities.append(out["v_theta"].detach().cpu())
            velocities = torch.cat(velocities, dim=0)
            if transport is not None:
                source_x_t, step_idx, endpoint_scale, endpoint_shift = transport
                velocities = transport_source_velocity_to_target(
                    source_x_t,
                    velocities,
                    step_idx,
                    int(self.cfg.actor.model.denoising_steps),
                    endpoint_scale,
                    endpoint_shift,
                )
            return velocities

        teacher_names = self._teacher_names()
        if not teacher_names:
            teacher_model = self._get_opd_teacher_model()
            teacher_model.to(self.device)
            teacher_velocities = _run(
                teacher_model, flat_forward_inputs, flat_batch_size
            )
            teacher_model.to("cpu")
            return teacher_velocities

        teacher_ids = flat_forward_inputs.get("teacher_ids", None)
        assert teacher_ids is not None, (
            "routed multi-teacher OPD needs per-sample teacher_ids in forward_inputs; "
            "the rollout worker attaches them when rollout.expert_models is set."
        )
        teacher_ids = teacher_ids.reshape(-1).cpu().long()
        assert teacher_ids.shape[0] == flat_batch_size, (
            f"teacher_ids has {teacher_ids.shape[0]} entries for {flat_batch_size} samples."
        )

        teacher_velocities = None
        counts = {}
        for teacher_idx in sorted(set(teacher_ids.tolist())):
            name = teacher_names[teacher_idx]
            mask = teacher_ids == teacher_idx
            rows = torch.nonzero(mask, as_tuple=False).reshape(-1)
            counts[name] = int(rows.numel())
            subset = {
                key: value[rows] if torch.is_tensor(value) else value
                for key, value in flat_forward_inputs.items()
            }
            teacher_model = self._get_routed_teacher_model(name)
            teacher_model.to(self.device)
            subset_velocities = _run(teacher_model, subset, int(rows.numel()))
            teacher_model.to("cpu")
            clear_memory()
            if teacher_velocities is None:
                teacher_velocities = torch.zeros(
                    (flat_batch_size, *subset_velocities.shape[1:]),
                    dtype=subset_velocities.dtype,
                )
            teacher_velocities[rows] = subset_velocities

        self.log_info(f"Routed teacher velocities over {counts}.")
        return teacher_velocities

    def _log_opd_reward_per_suite(
        self, teacher_logprobs: torch.Tensor, prev_logprobs: torch.Tensor
    ) -> None:
        """Break the OPD reward down by suite.

        ``r_OPD = l_T - l_S`` is M4's whole objective, and the aggregate says nothing about
        where it comes from. The three suites start from very different places -- the
        student scores 0.44 / 0.08 / 0.00 on the pilot tasks while the routed teachers
        score 0.85 / 0.75 / 0.52 -- so if the reward is the same size on the suite where
        the two policies are 0.52 apart as on the one where they are 0.42 apart, then this
        quantity is not tracking policy quality at all, and no learning rate will fix that.
        """
        teacher_ids = self.rollout_batch["forward_inputs"].get("teacher_ids", None)
        if teacher_ids is None:
            return
        names = self._teacher_names()
        ids, reward = select_opd_metric_rows(
            teacher_ids,
            teacher_logprobs,
            prev_logprobs,
            sample_idx=getattr(self, "_sample_idx", None),
            loss_mask=self.rollout_batch.get("loss_mask", None),
        )
        parts = {}
        for suite in sorted(set(ids.tolist())):
            rows = reward[ids == suite]
            parts[names[suite]] = (
                f"mean {rows.mean().item():+.4f} "
                f"sd {rows.std(unbiased=False).item():.4f}"
            )
        self.log_info(f"OPD reward per suite: {parts}")

    def _opd_prev_logprobs_at_sampled_step(
        self, step_idx: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """The student's log-prob of the transition it sampled, shaped for the advantage.

        ``compute_opd_advantages`` asserts the teacher's log-probs and the student's have
        the same shape, and it accepts ``[T, B, chunk, dim]``. The teacher's already are.
        The rollout's are not: they carry one entry per denoise step,
        ``[T, B, num_steps(+1), chunk, dim]``, because the rollout scores every step and
        leaves the deterministic ones at exactly 0.

        The reduced copy is stored under its own key rather than replacing
        ``prev_logprobs``. GR00T's ``default_forward`` re-derives its own aligned copy at
        training time with ``prev_logprobs[arange, denoise_inds[:, 0], ...]``, so
        collapsing that axis in place would break the indexing it depends on -- the two
        consumers want different shapes of the same quantity.

        Selection, not summation: the other steps are zero only because they are
        deterministic today.
        """
        prev = self.rollout_batch.get("prev_logprobs", None)
        if prev is None or prev.dim() < 5:
            # OpenVLA-style token log-probs already match the teacher's rank.
            return None
        return select_flow_opd_logprobs(
            prev,
            step_idx,
            joint_logprob=bool(self.cfg.actor.model.rl_head_config.joint_logprob),
        )

    def _teacher_transition_logprobs(
        self,
        flat_chains: torch.Tensor,
        x_cur: torch.Tensor,
        step_idx: torch.Tensor,
        teacher_velocities: torch.Tensor,
        num_steps: int,
        time_dim: int,
        batch_dim: int,
    ) -> torch.Tensor:
        """Method 4: teacher log-prob of the student's sampled Flow-SDE transition.

        The student sampled ``x_{k+1} ~ N(mu_S(x_k), Sigma_k)`` at denoise step ``k`` and
        the rollout stored ``l_S_old = log N(x_{k+1}; mu_S, Sigma_k)`` as ``prev_logprobs``.
        Scoring the *same* sample under the teacher's transition gives

            ``l_T = log N(x_{k+1}; mu_T(x_k), Sigma_k)``,  ``mu_T = x_k (w0 + w1) + v_T c``

        so ``r_OPD = l_T - l_S_old`` is the per-element reverse-KL reward. Elements are
        independent (diagonal Sigma), which is what makes the per-element score-function
        estimator unbiased and lower-variance than a summed one.

        Returns:
            ``[T, B, action_chunk, action_env_dim]``, matching ``prev_logprobs``.
        """
        noise_level, chunk, env_dim, c_sign = self._flow_sde_transition_spec()

        return flow_teacher_transition_logprobs(
            flat_chains=flat_chains,
            x_cur=x_cur,
            step_idx=step_idx,
            teacher_velocities=teacher_velocities,
            num_steps=num_steps,
            noise_level=noise_level,
            action_chunk=chunk,
            action_env_dim=env_dim,
            c_sign=c_sign,
            time_dim=time_dim,
            batch_dim=batch_dim,
        )

    def _build_sft_data_loader(self):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            repo_id = resolve_lerobot_repo_id(self.cfg.actor.get("sft_data_path"))
            if repo_id is None:
                raise ValueError(
                    "actor.sft_data_path must be set to a local dataset path or "
                    "LeRobot repo id when enable_sft_co_train=True."
                )

            import openpi.training.data_loader as _data

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            if "config_name" not in self.cfg.actor:
                raise ValueError(
                    "config_name is required when enable_sft_co_train=True"
                )
            training_config_name = self.cfg.actor.config_name
            data_loader_config = get_openpi_config(
                training_config_name,
                model_path=self.cfg.actor.model.model_path,
                repo_id=repo_id,
                data_kwargs=getattr(self.cfg.actor.model, "openpi_data", None),
            )
            self.data_loader = _data.create_data_loader(
                data_loader_config, framework="pytorch", shuffle=True
            )
            self.sft_iterator = iter(self.data_loader)
            self.train_epoch = 0
            self.sft_loss_weight = self.cfg.actor.get("sft_loss_weight", 0.1)
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _train_sft_epoch(
        self, metrics_data: dict[str, torch.Tensor], loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Train one epoch of SFT.
        """
        metrics_data["ppo_loss"] = loss.clone().detach().item()

        # Get next data batch
        try:
            observation, actions = next(self.sft_iterator)
        except StopIteration:
            self.train_epoch += 1
            self.data_loader.set_epoch(self.train_epoch)
            self.sft_iterator = iter(self.data_loader)
            observation, actions = next(self.sft_iterator)

        sft_loss = self.model(
            data=(observation, actions),
            forward_type=ForwardType.SFT,
        )
        metrics_data["sft_loss"] = sft_loss.detach().item()
        total_loss = loss + self.sft_loss_weight * sft_loss
        loss = total_loss

        metrics_data["loss_ratio"] = (
            np.abs(metrics_data["sft_loss"]) / np.abs(metrics_data["ppo_loss"])
            if np.abs(metrics_data["ppo_loss"]) > 0
            else float("inf")
        )
        if metrics_data["loss_ratio"] > 1e5:
            self.logger.warning(
                "SFT/PPO loss imbalance detected: "
                f"ratio={metrics_data['loss_ratio']:.3e}, "
                f"sft_loss={metrics_data['sft_loss']:.6f}, "
                f"ppo_loss={metrics_data['ppo_loss']:.6f}, "
                f"sft_loss_weight={self.sft_loss_weight:.6f}"
            )
        return loss

    @Worker.timer("run_training")
    def run_training(self) -> dict[str, float]:
        """Train once, or refresh flow between updates on the latest observation pool."""
        repeats = int(self.cfg.algorithm.get("flow_refresh_updates", 1))
        update_steps = self.cfg.get("runner", {}).get("step_unit") == "optimizer_update"
        if repeats == 1 and not update_steps:
            return self._run_training_once()

        # _run_training_once replaces self.rollout_batch with a sampled, flat batch.
        # Keep the full CPU rollout alive only until the last inner update finishes.
        observation_pool = self.rollout_batch
        # Count executed controls before collapsing to chunk-level training masks.
        control_mask, _ = compute_loss_mask(observation_pool["dones"])
        interaction_metrics = all_reduce_dict(
            {"refresh/real_env_steps_this_rollout": float(control_mask.sum())},
            op=torch.distributed.ReduceOp.SUM,
        )
        del control_mask
        results = []
        metrics = dict(interaction_metrics)
        for inner in range(repeats):
            self.log_info(
                f"Flow refresh: rollout {self.version + 1}, update {inner + 1}/{repeats}, "
                f"source={'rollout' if inner == 0 else 'current_student'}."
            )
            if inner:
                self.rollout_batch = observation_pool
                refresh_start = time.perf_counter()
                self._prepare_flow_refresh_batch()
                metrics[f"refresh/update_{inner + 1}/prepare_seconds"] = (
                    time.perf_counter() - refresh_start
                )
            update_start = time.perf_counter()
            previous_updates = getattr(self, "optimizer_steps", 0)
            result = self._run_training_once()
            if update_steps and self.optimizer_steps != previous_updates + 1:
                raise RuntimeError(
                    "Each controlled inner update must call optimizer once."
                )
            metrics[f"refresh/update_{inner + 1}/train_seconds"] = (
                time.perf_counter() - update_start
            )
            results.append(result)
            metrics.update(
                {
                    f"refresh/update_{inner + 1}/{key}": value
                    for key, value in result.items()
                }
            )
            self.log_info(
                f"Flow refresh update {inner + 1}/{repeats} complete: "
                f"optimizer_updates={self.lr_scheduler.last_epoch}, metrics={result}."
            )
        metrics.update(
            {key: float(np.mean([r[key] for r in results])) for key in results[0]}
        )
        metrics["refresh/updates_per_rollout"] = repeats
        metrics["refresh/optimizer_updates"] = self.lr_scheduler.last_epoch
        metrics["refresh/physical_rollouts"] = self.version + 1
        return metrics

    @torch.no_grad()
    def _prepare_flow_refresh_batch(self) -> None:
        """Resample observations, generate current flow, and query routed teachers.

        Duplicated observations get independent flow draws. Keep them as distinct
        rows throughout generation and supervision, rather than scattering fresh
        flow states back to a possibly repeated observation index.
        """
        time_dim, batch_dim = self.rollout_batch["loss_mask"].shape[:2]
        sample_idx = self._suite_balanced_sample(
            time_dim * batch_dim, time_dim, batch_dim
        )
        assert sample_idx is not None, "Flow refresh requires suite-balanced sampling."
        keys = ("forward_inputs", "prev_logprobs", "loss_mask", "loss_mask_sum")
        selected = process_nested_dict_for_train(
            {key: self.rollout_batch[key] for key in keys}, sample_idx
        )
        inputs = selected["forward_inputs"]
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        self.model.eval()
        refresh_start = time.perf_counter()
        chains = []
        size = int(sample_idx.numel())
        assert size % self.cfg.actor.micro_batch_size == 0
        isolated_rng = self.cfg.get("runner", {}).get("step_unit") == "optimizer_update"
        devices = []
        if isolated_rng:
            # Worker CUDA platforms return an integer; CPU tests use torch.device.
            device = (
                torch.device("cuda", self.device)
                if isinstance(self.device, int)
                else torch.device(self.device)
            )
            if device.type == "cuda":
                devices = [
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                ]
        rng_context = (
            torch.random.fork_rng(devices=devices) if isolated_rng else nullcontext()
        )
        with rng_context:
            if isolated_rng:
                seed = (
                    self.cfg.actor.seed + self._rank + self._sample_draw_count * 104729
                )
                torch.random.default_generator.manual_seed(seed)
                for device_index in devices:
                    torch.cuda.default_generators[device_index].manual_seed(seed)
            for micro in split_dict_to_chunk(
                inputs, size // self.cfg.actor.micro_batch_size
            ):
                micro = put_tensor_device(micro, self.device)
                with self.amp_context:
                    output = self.model(
                        forward_type=ForwardType.FLOW_REFRESH,
                        forward_inputs=micro,
                    )
                chains.append(output["chains"].detach().cpu())
        inputs["chains"] = torch.cat(chains, dim=0)
        self.log_info(
            f"Refreshed {size} sampled flows in "
            f"{time.perf_counter() - refresh_start:.2f}s on rank {self._rank}."
        )
        del micro, output, chains
        if self.enable_offload:
            self.offload_param_and_grad()
        clear_memory()

        # Reuse the original teacher-coordinate adapter, timestep sampler and loss
        # preparation. The singleton time axis represents independent sampled rows;
        # it does not describe new physical transitions or a new environment rollout.
        self.rollout_batch = {
            key: (
                {name: value.unsqueeze(0) for name, value in item.items()}
                if isinstance(item, dict)
                else item.unsqueeze(0)
            )
            for key, item in selected.items()
        }
        self.compute_teacher_velocities(already_sampled=True)

    def _run_training_once(self) -> dict[str, float]:
        """
        Run the training process using the received rollout batch.
        """
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        if self.cfg.algorithm.loss_type == "opd":
            target_steps = int(self.rollout_batch["advantages"].shape[0])
            for key in [
                "prev_logprobs",
                "forward_inputs",
                "loss_mask",
                "loss_mask_sum",
            ]:
                assert key in self.rollout_batch, f"OPD training requires {key}."
                self.rollout_batch[key] = trim_nested_tensor_time_dim(
                    self.rollout_batch[key], target_steps, (key,)
                )

        if self.cfg.algorithm.loss_type in VELOCITY_DISTILL_LOSS_TYPES:
            assert "loss_mask" in self.rollout_batch, (
                "velocity distillation requires loss_mask."
            )
            target_steps = int(self.rollout_batch["loss_mask"].shape[0])
            keys = [
                "prev_logprobs",
                "forward_inputs",
                "nft_x_t",
                "nft_timesteps",
                "teacher_velocities",
                "loss_mask",
                "loss_mask_sum",
            ]
            if self.cfg.algorithm.loss_type == "analytic_local_kl":
                keys.append("nft_kl_weight")
            for key in keys:
                assert key in self.rollout_batch, (
                    f"velocity distillation requires {key}."
                )
                self.rollout_batch[key] = trim_nested_tensor_time_dim(
                    self.rollout_batch[key], target_steps, (key,)
                )

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        # compute_teacher_velocities drew a suite-balanced subset and scored the teachers
        # only on it; reuse exactly those indices so the student is not run on transitions
        # the loss will ignore. Both sites flatten [T, B] the same way
        # (value.reshape(-1, ...)), so a flat index means the same (t, b) in each.
        sample_idx = getattr(self, "_sample_idx", None)
        self._sample_idx = None  # consumed; a stale index must never be reused
        if sample_idx is not None:
            shuffle_id = sample_idx
        else:
            g = torch.Generator()
            g.manual_seed(self.cfg.actor.seed + self._rank)
            shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, batch in enumerate(train_micro_batch):
                    self.train_micro_batch(
                        micro_batch=batch,
                        metrics=metrics,
                        is_last=(idx + 1) == self.gradient_accumulation,
                    )
                    # avoid gpu memory leak
                    train_micro_batch[idx] = None
                    del batch

                self.torch_platform.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                if self.cfg.runner.get(
                    "step_unit"
                ) == "optimizer_update" and not torch.isfinite(
                    torch.as_tensor(grad_norm)
                ):
                    raise RuntimeError(
                        "Non-finite gradient skipped an optimizer update; stopping "
                        "instead of advancing the controlled experiment's step."
                    )
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        explained_variance_stats = pop_critic_explained_variance_stats(metrics)
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        if explained_variance_stats:
            reduced_stats = all_reduce_dict(
                explained_variance_stats, op=torch.distributed.ReduceOp.SUM
            )
            mean_metric_dict[CRITIC_EXPLAINED_VARIANCE_KEY] = (
                compute_critic_explained_variance_from_stats(reduced_stats).item()
            )

        return mean_metric_dict

    def train_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        metrics: dict[str, list[float]],
        *,
        is_last: bool,
    ) -> None:
        micro_batch = put_tensor_device(micro_batch, self.device)
        backward_ctx = self.before_micro_batch(self.model, is_last_micro_batch=is_last)
        advantages = micro_batch.get("advantages", None)  # absent for velocity_distill
        prev_logprobs = micro_batch["prev_logprobs"]
        returns = micro_batch.get("returns", None)
        prev_values = micro_batch.get("prev_values", None)
        loss_mask = micro_batch.get("loss_mask", None)
        loss_mask_sum = micro_batch.get("loss_mask_sum", None)
        forward_inputs = micro_batch.get("forward_inputs", None)

        if self.cfg.algorithm.loss_type in VELOCITY_DISTILL_LOSS_TYPES:
            # Method 1 / 3: student velocity v_S at the SAME (x_t, tau, o_t) the frozen
            # teacher was scored on; supervised match against the cached teacher velocity
            # (uniform for velocity_distill, w(tau)-weighted for analytic_local_kl).
            with self.amp_context:
                student_out = self.model(
                    forward_type=ForwardType.NFT,
                    forward_inputs=forward_inputs,
                    nft_inputs={
                        "x_t": micro_batch["nft_x_t"],
                        "timesteps": micro_batch["nft_timesteps"],
                    },
                    compute_values=False,
                )
            # Call the registered loss directly: these are pure supervised losses and must
            # skip the PPO/OPD embodied preprocessing (logprobs/advantages).
            loss_kwargs = {
                "student_velocity": student_out["v_theta"],
                "teacher_velocity": micro_batch["teacher_velocities"],
                "loss_mask": loss_mask,
            }
            if self.cfg.algorithm.loss_type == "analytic_local_kl":
                loss_kwargs["kl_weight"] = micro_batch["nft_kl_weight"]
            loss, metrics_data = get_policy_loss(self.cfg.algorithm.loss_type)(
                **loss_kwargs
            )
            loss = loss / self.gradient_accumulation
            with backward_ctx:
                self.grad_scaler.scale(loss).backward()
            metrics_data["actor/total_loss"] = loss.detach().item()
            append_to_dict(metrics, metrics_data)
            return

        kwargs = {}
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            kwargs["temperature"] = self.cfg.rollout.sampling_params.temperature_train
            kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            kwargs["prev_logprobs"] = prev_logprobs

        compute_values = self.cfg.algorithm.adv_type == "gae"
        with self.amp_context:
            output_dict = self.model(
                forward_inputs=forward_inputs,
                compute_logprobs=True,
                compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                compute_values=compute_values,
                use_cache=False,
                **kwargs,
            )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            prev_logprobs = output_dict["prev_logprobs"]

        loss_kwargs = {
            "loss_type": self.cfg.algorithm.loss_type,
            "logprob_type": self.cfg.algorithm.logprob_type,
            "reward_type": self.cfg.algorithm.reward_type,
            "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
            "logprobs": output_dict["logprobs"],
            "values": output_dict.get("values", None),
            "old_logprobs": prev_logprobs,
            "advantages": advantages,
            "returns": returns,
            "prev_values": prev_values,
            "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
            "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
            "value_clip": self.cfg.algorithm.get("value_clip", None),
            "huber_delta": self.cfg.algorithm.get("huber_delta", None),
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "max_episode_steps": self.cfg.env.train.max_episode_steps,
            "task_type": self.cfg.runner.task_type,
            "critic_warmup": self.optimizer_steps < self.critic_warmup_steps,
        }

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
        ]:
            loss_kwargs["clip_ratio_c"] = self.cfg.algorithm.get("clip_ratio_c", 3.0)
            if self.cfg.algorithm.get("clip_log_ratio_min") is not None:
                loss_kwargs["clip_log_ratio_min"] = (
                    self.cfg.algorithm.clip_log_ratio_min
                )
            if self.cfg.algorithm.get("clip_log_ratio_max") is not None:
                loss_kwargs["clip_log_ratio_max"] = (
                    self.cfg.algorithm.clip_log_ratio_max
                )

        loss, metrics_data = policy_loss(**loss_kwargs)
        entropy_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
        if self.cfg.algorithm.entropy_bonus > 0 and not loss_kwargs["critic_warmup"]:
            entropy = output_dict["entropy"]
            entropy = reshape_entropy(
                entropy,
                entropy_type=self.cfg.algorithm.entropy_type,
                action_dim=self.cfg.actor.model.get("action_dim", 7),
                batch_size=output_dict["logprobs"].shape[0],
            )
            entropy_loss = masked_mean(entropy, mask=loss_mask)
            loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
        metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()

        if self.enable_sft_co_train:
            loss = self._train_sft_epoch(metrics_data, loss)

        loss /= self.gradient_accumulation
        with backward_ctx:
            self.grad_scaler.scale(loss).backward()

        metrics_data["actor/total_loss"] = loss.detach().item()
        append_to_dict(metrics, metrics_data)

    def restore_opd_update_counters(self, expected_updates: int) -> dict:
        """Reconcile actor counters with a loaded adaptive-round DCP checkpoint."""
        if self.cfg.runner.get("resume_mode") != "adaptive_round_boundary":
            raise ValueError("Explicit adaptive_round_boundary resume mode required.")
        if self.lr_scheduler.last_epoch != expected_updates or not self.optimizer.state:
            raise ValueError(
                "Checkpoint optimizer/scheduler state is missing or mismatched."
            )
        self.optimizer_steps = expected_updates
        # This recipe makes one suite-balanced draw per optimizer update.
        self._sample_draw_count = expected_updates
        result = {
            "rank": self._rank,
            "optimizer_updates": self.optimizer_steps,
            "scheduler_last_epoch": self.lr_scheduler.last_epoch,
            "sample_draw_count": self._sample_draw_count,
            "lr": self.optimizer.param_groups[0]["lr"],
            "optimizer_state_entries": len(self.optimizer.state),
        }
        self.log_info(f"Restored OPD training state: {result}")
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()
            if not self.is_weight_offloaded:
                self.offload_param_and_grad()
        return result

    def set_global_step(self, global_step: int) -> None:
        """
        Set the global step for the model, if needed.
        """
        self.version = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def finish_global_batch(self, metrics: dict[str, list[float]]) -> None:
        self.torch_platform.empty_cache()
        grad_norm, lr_list = self.optimizer_step()
        self.optimizer.zero_grad()
        metric_data = {
            "actor/grad_norm": grad_norm,
            "actor/lr": lr_list[0],
        }
        if len(lr_list) > 1:
            metric_data["critic/lr"] = lr_list[1]
        append_to_dict(metrics, metric_data)
