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
import random
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import torch
from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.state_action import StateActionTransform
from gr00t.model.action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from gr00t.model.gr00t_n1 import GR00T_N1_5, GR00T_N1_5_Config
from torch.distributions import Normal
from transformers.feature_extraction_utils import BatchFeature

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.gr00t.simulation_io import (
    ACTION_CONVERSION_N1D5,
    OBS_CONVERSION,
    ROBOCASA_STATE_SLICES,
)
from rlinf.models.embodiment.gr00t.utils import (
    squeeze_dict_values,
    unsqueeze_dict_values,
)
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead


class FlowMatchingActionHeadForRLActionPrediction(FlowmatchingActionHead):
    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
        rl_head_config: dict[str, Any],
        output_action_chunks: int,
        valid_action_dim: int,
    ):
        super().__init__(config)
        self.action_chunk = output_action_chunks
        self.rl_config = rl_head_config
        self.padding_value = rl_head_config.padding_value
        self.valid_action_dim = valid_action_dim

        if self.rl_config.use_vlm_value:
            proj_width = 2048
        else:
            proj_width = 3584

        if self.rl_config.add_value_head:
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=(1024, 512, 256),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )

        if self.rl_config.noise_method == "reinflow":
            self.reinflow_explore_noise_net = ExploreNoiseNet(
                in_dim=self.hidden_size,
                out_dim=self.config.action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=[0.08, 0.16],
                noise_scheduler_type="learn",
            )

    @staticmethod
    def _cast_batch_feature_floats(
        feature: BatchFeature, dtype: torch.dtype
    ) -> BatchFeature:
        """Cast floating tensors at the nested FSDP action-head boundary.

        FSDP's ``cast_forward_inputs`` recursively handles standard Python containers but
        not Transformers' ``BatchFeature``. GR00T prepares inputs before entering the
        nested action-head FSDP wrapper, when ``action_head.dtype`` still reports the FP32
        master dtype. Once the wrapper unshards for compute, its parameters are BF16, so
        leaving the nested state tensor in FP32 makes the first category-specific matrix
        multiply fail. At this point the wrapper's parameter views already expose the
        actual compute dtype, making it the authoritative target.
        """
        return BatchFeature(
            data={
                key: (
                    value.to(dtype=dtype)
                    if isinstance(value, torch.Tensor) and value.is_floating_point()
                    else value
                )
                for key, value in feature.items()
            }
        )

    def get_logprob_norm(self, sample, mu, sigma):
        if self.rl_config.safe_get_logprob:
            dist = Normal(loc=mu, scale=sigma)
            return dist.log_prob(sample)
        else:
            # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
            return log_prob

    def get_velocity(
        self,
        vl_embs: torch.Tensor,
        x_t: torch.Tensor,
        idx: torch.Tensor,
        denoise_steps: int,
        embodiment_id: torch.Tensor,
        state_features: torch.Tensor,
    ):
        """Predict the flow velocity v_theta at an explicit (x_t, idx) pair.

        Split out of :meth:`sample_mean_var_val` so that velocity distillation can
        score a teacher and a student on exactly the same computation. ``idx`` is the
        integer denoise index; GR00T's continuous time is ``idx / denoise_steps`` and
        runs from 0 (noise) to 1 (data), the opposite of OpenPI's convention.

        Returns:
            The velocity ``v_t`` and the raw ``model_output`` (kept for the value head).
        """
        device = vl_embs.device
        t_cont = idx / float(denoise_steps)
        timesteps_tensor = (
            (t_cont * self.num_timestep_buckets).to(torch.int64).to(device)
        )
        action_features = self.action_encoder(x_t, timesteps_tensor, embodiment_id)
        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=device
            )
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
            vl_embs.shape[0], -1, -1
        )
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
        )
        model_output = model_output[:, -self.action_horizon :]

        v_t = self.action_decoder(model_output, embodiment_id)
        return v_t, model_output

    def sample_mean_var_val(
        self,
        vl_embs: torch.Tensor,
        denoise_steps: int,
        x_t: torch.Tensor,
        embodiment_id: int,
        state_features: torch.Tensor,
        idx: Optional[int | torch.Tensor],
        mode: Literal["train", "eval"] = "train",
        compute_values=False,
    ):
        """
        Sample the mean, variance and value of the action at a given timestep.
        Rollout sample (idx is int) and actor get_log_prob_value (idx is tensor) will load this function.
        Pay attention: The time notation of gr00t is different from openpi.
        In gr00t, the time is from 0 to 1, while in openpi, the time is from 1 to 0.
        """
        # expand the shape
        bsize = vl_embs.shape[0]
        device = vl_embs.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)
        # build parameters
        if self.rl_config.noise_anneal:
            # noise annealing
            noise_start, noise_end, anneal_steps = self.rl_config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            # fixed noise level
            noise_level = torch.tensor(self.rl_config.noise_level).to(device)

        # velocity prediction
        v_t, model_output = self.get_velocity(
            vl_embs=vl_embs,
            x_t=x_t,
            idx=idx,
            denoise_steps=denoise_steps,
            embodiment_id=embodiment_id,
            state_features=state_features,
        )

        timesteps = torch.linspace(
            0, 1, denoise_steps + 1, device=device, dtype=vl_embs.dtype
        )
        t_input = timesteps[idx]
        delta = timesteps[idx + 1] - timesteps[idx]
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        # Emphasize: In Gr00t, x0: noise, x1: data.
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if mode == "eval":
            x0_weight = 1 - (t_input + delta)
            x1_weight = (
                t_input + delta
            )  # notice the plus here, it's different from openpi.
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.rl_config.noise_method == "flow_ode":
                # Same Euler step as the mode == "eval" branch above. It is duplicated
                # because the two are selected by different things: `mode` is chosen per
                # denoise step, `noise_method` is fixed for the run -- so without this
                # there is no way to ask for a rollout where no step is stochastic.
                x0_weight = 1 - (t_input + delta)
                x1_weight = t_input + delta
                x_t_std = torch.zeros_like(t_input)
            elif self.rl_config.noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        (1 - timesteps)
                        / torch.where(timesteps == 0, timesteps[1], timesteps)
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = (
                    torch.ones_like(t_input)
                    - (t_input + delta)
                    - sigma_i**2 * delta / (2 * (1 - t_input))
                )
                x1_weight = t_input + delta
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.rl_config.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = (torch.ones_like(t_input) - (t_input + delta)) * cos_term
                x1_weight = t_input + delta
                x_t_std = (1 - (t_input + delta)) * sin_term
            elif self.rl_config.noise_method == "reinflow":
                x0_weight = 1 - (t_input + delta)
                x1_weight = t_input + delta
                x_t_std = self.reinflow_explore_noise_net(model_output)
            else:
                raise ValueError(
                    f"Invalid noise method: {self.rl_config.noise_method}. "
                    'Supported: "flow_ode", "flow_sde", "flow_cps", "reinflow".'
                )
        # In eval, this equals to x_t_mean = x_t + v*dt(dt>0).
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std

    def get_velocity_from_backbone(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        x_t: torch.Tensor,
        idx: torch.Tensor,
        denoise_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Velocity at an explicit (x_t, idx), starting from a backbone forward.

        Used by velocity distillation, where a frozen teacher and the student must be
        scored at exactly the same flow state.
        """
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        state_features = self.state_encoder(action_input.state, embodiment_id)
        if denoise_steps is None:
            denoise_steps = self.num_inference_timesteps
        v_t, _ = self.get_velocity(
            vl_embs=vl_embs,
            x_t=x_t.to(vl_embs.dtype),
            idx=idx,
            denoise_steps=denoise_steps,
            embodiment_id=embodiment_id,
            state_features=state_features,
        )
        return v_t

    def get_rl_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
    ) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)
        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        x_t = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        chains = [x_t]
        log_probs = []

        if self.rl_config.joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(x_t), torch.ones_like(x_t)
            )
            log_probs.append(initial_log_prob)

        num_steps = self.num_inference_timesteps
        # determine the denoise step for the logprob calculation
        if mode == "train":
            if self.rl_config.joint_logprob:
                # Every step takes the mode="train" branch -- deterministic under
                # flow_ode (M1/M3), perturbed under a stochastic noise_method.
                denoise_inds = torch.arange(num_steps)
            else:
                if self.rl_config.noise_method == "flow_ode":
                    # No stochastic transition to sample, and denoise_inds decides which
                    # flow state teacher supervision scores -- so this is an error, not a no-op.
                    raise ValueError(
                        'noise_method="flow_ode" requires joint_logprob=True: a '
                        "deterministic rollout has no stochastic step for "
                        "joint_logprob=False to select."
                    )
                elif self.rl_config.noise_method == "flow_sde":
                    if self.rl_config.ignore_last:
                        denoise_inds = torch.tensor(
                            [random.randint(0, num_steps - 2)] * num_steps
                        )
                    else:
                        denoise_inds = torch.tensor(
                            [random.randint(0, num_steps - 1)] * num_steps
                        )
                elif self.rl_config.noise_method == "flow_cps":
                    # the last denoising step of the flow-cps is deterministic
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
                elif self.rl_config.noise_method == "reinflow":
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(batch_size, 1)

        # Run denoising steps.
        for idx in range(num_steps):
            if idx == denoise_inds[0][idx]:
                x_t_mean, x_t_std = self.sample_mean_var_val(
                    vl_embs=vl_embs,
                    idx=idx,
                    x_t=x_t,
                    embodiment_id=embodiment_id,
                    state_features=state_features,
                    mode="train",
                    denoise_steps=num_steps,
                    compute_values=compute_values,
                )
            else:
                x_t_mean, x_t_std = self.sample_mean_var_val(
                    vl_embs=vl_embs,
                    idx=idx,
                    x_t=x_t,
                    embodiment_id=embodiment_id,
                    state_features=state_features,
                    mode="eval",
                    denoise_steps=num_steps,
                    compute_values=compute_values,
                )

            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)

            chains.append(x_t)
            log_probs.append(log_prob)

        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.action_chunk, : self.valid_action_dim
        ]
        # Velocity distillation runs without a critic, so the value head may not exist;
        # asking for values then is a config error, not something to crash on.
        if compute_values and self.rl_config.add_value_head:
            values = self.get_value(vl_embs, state_features)
            values = values[:, None]
        else:
            values = torch.zeros((batch_size, 1), device=device, dtype=vl_embs.dtype)

        return BatchFeature(
            data={"action_pred": x_0}
        ), {  # this is for gr00t validity check
            "actions": x_0,
            "action_pred": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    def forward(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        chains=None,
        denoise_inds=None,
        compute_values=True,
        nft_inputs=None,
        refresh_flow: bool = False,
    ):
        compute_dtype = self.state_encoder.layer1.W.dtype
        backbone_output = self._cast_batch_feature_floats(
            backbone_output, compute_dtype
        )
        action_input = self._cast_batch_feature_floats(action_input, compute_dtype)
        if refresh_flow:
            # Enter through __call__ so the nested FSDP head is unsharded.
            _, outputs = self.get_rl_action(
                backbone_output, action_input, mode="train", compute_values=False
            )
            return outputs["chains"]
        if isinstance(chains, torch.Tensor) and chains.is_floating_point():
            chains = chains.to(dtype=compute_dtype)
        if nft_inputs is not None:
            nft_inputs = {
                key: (
                    value.to(dtype=compute_dtype)
                    if isinstance(value, torch.Tensor) and value.is_floating_point()
                    else value
                )
                for key, value in nft_inputs.items()
            }
        if nft_inputs is not None:
            # Velocity distillation. This must be reached through __call__ so that FSDP
            # all-gathers the head's parameters: calling get_velocity_from_backbone()
            # directly on an FSDP-wrapped head skips the hook and the first LayerNorm
            # sees a zero-sized weight.
            return self.get_velocity_from_backbone(
                backbone_output=backbone_output,
                action_input=action_input,
                x_t=nft_inputs["x_t"],
                idx=nft_inputs["timesteps"],
            )
        backbone_output = self.process_backbone_output(backbone_output)
        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]

        chains_log_probs = []

        if self.rl_config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            chains_log_probs.append(initial_log_prob)
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(batch_size), denoise_ind]
            chains_next = chains[torch.arange(batch_size), denoise_ind + 1]
            x_t_mean, x_t_std = self.sample_mean_var_val(
                vl_embs=vl_embs,
                idx=denoise_ind,
                x_t=chains_pre,
                embodiment_id=embodiment_id,
                state_features=state_features,
                mode="train",
                denoise_steps=self.num_inference_timesteps,
                compute_values=compute_values,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            chains_log_probs.append(log_probs)

        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        if compute_values and self.rl_config.add_value_head:
            chains_values = self.get_value(vl_embs, state_features)
            chains_values = chains_values[:, None]
        else:
            chains_values = torch.zeros(
                (batch_size, 1), device=chains_log_probs.device, dtype=vl_embs.dtype
            )  # (B, 1)
        return chains_log_probs, chains_values

    def sample_noise(self, shape, device):
        sample_dtype = (
            torch.float32 if self._use_fp32_normal(device) else torch.bfloat16
        )
        result = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=sample_dtype,
            device=device,
        )
        return result.to(torch.bfloat16)

    @staticmethod
    def _use_fp32_normal(device):
        device_type = torch.device(device).type
        return device_type == "npu"

    def get_value(self, vl_embs, state_features):
        # TODO: add value vlm mode param
        bsize = vl_embs.shape[0]
        mask_length = vl_embs.shape[1]
        if self.rl_config.value_vlm_mode == "mean_token":
            prefix_mask = [True] * mask_length
        elif self.rl_config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (mask_length - 1) + [True] * 1
        elif self.rl_config.value_vlm_mode == "first_token":
            prefix_mask = [True] * 1 + [False] * (mask_length - 1)
        vl_embs_value = vl_embs[:, prefix_mask, :]
        vl_embs_value = vl_embs_value.mean(dim=1, keepdim=False)
        # vl_embs_value = vl_embs_value.to(dtype=torch.float32)
        state_features_value = state_features.reshape(bsize, -1)
        if self.rl_config.use_vlm_value:
            value_embs = vl_embs_value
        else:
            value_embs = torch.cat((vl_embs_value, state_features_value), dim=1)
        values_vlm = self.value_head(value_embs)[:, 0]
        return values_vlm


class GR00T_N1_5_ForRLActionPrediction(GR00T_N1_5, BasePolicy):
    """
    GR00T_N1_5 model for reinforcement learning action prediction.
    It's a combination of the Gr00tPolicy and GR00T_N1_5 model.

    Notes:
        - Device is handled by huggingface worker.
        - EmbodimentTag determines the state encoder and action head to use.
          we use "new_embodiment" reserved by gr00t.

    """

    _no_split_modules = [
        "Eagle2_5_VLForConditionalGeneration",
        "FlowMatchingActionHeadForRLActionPrediction",
        "TimestepEncoder",
        "TimestepEmbedding",
        "ValueHead",
    ]

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        rl_head_config: dict[str, Any],
        local_model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        compute_dtype: torch.dtype = torch.bfloat16,
        denoising_steps: Optional[int] = None,
        obs_converter_type: str = "libero",
        output_action_chunks: int = 1,
    ):
        super().__init__(config, local_model_path)

        self.padding_value = rl_head_config.padding_value
        self._modality_config = modality_config  # ModalityConfig(delta_indices=[0], modality_keys=['video.ego_view'])
        self._modality_transform = modality_transform
        self.model_path = Path(local_model_path)
        self.compute_dtype = compute_dtype
        self.output_action_chunks = output_action_chunks
        self.model_path = Path(local_model_path)

        # Convert string embodiment tag to EmbodimentTag enum if needed
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        if denoising_steps is not None:
            if hasattr(self, "action_head") and hasattr(
                self.action_head, "num_inference_timesteps"
            ):
                self.action_head.num_inference_timesteps = denoising_steps

        self.obs_convert_fn = OBS_CONVERSION[obs_converter_type]
        self.action_convert_fn = ACTION_CONVERSION_N1D5[obs_converter_type]
        self._opd_state_adapter_validated = False
        self._load_metadata(self.model_path / "experiment_cfg")

        # The param loading is after construction in from_pretrained(), so it should be safe to to so.
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowMatchingActionHeadForRLActionPrediction(
            action_head_cfg, rl_head_config, output_action_chunks, self.valid_action_dim
        )

    def eval(self):
        self._modality_transform.eval()
        return super().eval()

    def _check_state_is_batched(self, obs: dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        elif forward_type == ForwardType.NFT:
            return self.nft_forward(**kwargs)
        elif forward_type == ForwardType.FLOW_REFRESH:
            return self.nft_forward(refresh_flow=True, **kwargs)
        else:
            raise NotImplementedError

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = True,
        use_cache: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        normalized_input = {
            "state": forward_inputs["state"],
            "state_mask": forward_inputs["state_mask"],
            "eagle_input_ids": forward_inputs["eagle_input_ids"],
            "eagle_attention_mask": forward_inputs["eagle_attention_mask"],
            "eagle_pixel_values": forward_inputs["eagle_pixel_values"].reshape(
                -1, *forward_inputs["eagle_pixel_values"].shape[2:]
            ),
            "eagle_image_sizes": forward_inputs["eagle_image_sizes"].reshape(
                -1, *forward_inputs["eagle_image_sizes"].shape[2:]
            ),
            "embodiment_id": forward_inputs["embodiment_id"],
        }
        backbone_inputs, action_inputs = self.prepare_input(normalized_input)
        backbone_outputs = self.backbone(backbone_inputs)

        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]
        log_probs, value_t = self.action_head(
            backbone_output=backbone_outputs,
            action_input=action_inputs,
            chains=chains,
            denoise_inds=denoise_inds,
            compute_values=compute_values,
        )

        log_probs = log_probs[
            :,
            :,
            : self.action_head.action_chunk,
            : self.valid_action_dim,
        ]
        # post process
        if self.action_head.rl_config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
            prev_logprobs = kwargs["prev_logprobs"].mean(dim=1)
        else:
            bsize = log_probs.shape[0]
            log_probs = log_probs[:, 0]
            prev_logprobs = kwargs["prev_logprobs"]
            prev_logprobs = prev_logprobs[
                torch.arange(bsize),
                denoise_inds[:, 0],
                : self.action_head.action_chunk,
                : self.valid_action_dim,
            ]
        value_t = value_t.mean(dim=-1, keepdim=False)

        return {
            "logprobs": log_probs.float(),
            "prev_logprobs": prev_logprobs.float(),
            "values": value_t,
            "entropy": None,
        }

    def nft_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        refresh_flow: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Compute velocity v_theta at an explicit (x_t, denoise index) for NFT losses.

        ``kwargs["nft_inputs"]`` carries ``x_t`` (the flow state, ``[B, horizon, dim]``)
        and ``timesteps`` (the integer denoise index ``k``, ``[B]``). The returned
        velocity is sliced to the executed action chunk, matching how log-probs are
        sliced elsewhere in this class.
        """
        normalized_input = {
            "state": forward_inputs["state"],
            "state_mask": forward_inputs["state_mask"],
            "eagle_input_ids": forward_inputs["eagle_input_ids"],
            "eagle_attention_mask": forward_inputs["eagle_attention_mask"],
            "eagle_pixel_values": forward_inputs["eagle_pixel_values"].reshape(
                -1, *forward_inputs["eagle_pixel_values"].shape[2:]
            ),
            "eagle_image_sizes": forward_inputs["eagle_image_sizes"].reshape(
                -1, *forward_inputs["eagle_image_sizes"].shape[2:]
            ),
            "embodiment_id": forward_inputs["embodiment_id"],
        }
        backbone_inputs, action_inputs = self.prepare_input(normalized_input)
        backbone_outputs = self.backbone(backbone_inputs)

        if refresh_flow:
            chains = self.action_head(
                backbone_output=backbone_outputs,
                action_input=action_inputs,
                refresh_flow=True,
                compute_values=False,
            )
            return {"chains": chains}

        nft_inputs = kwargs["nft_inputs"]
        device = next(self.parameters()).device
        x_t = nft_inputs["x_t"].to(device)
        idx = nft_inputs["timesteps"].to(device)

        v_theta = self.action_head(
            backbone_output=backbone_outputs,
            action_input=action_inputs,
            nft_inputs={"x_t": x_t, "timesteps": idx},
        )
        v_theta = v_theta[:, : self.action_head.action_chunk, : self.valid_action_dim]
        return {"v_theta": v_theta, "x_t": x_t, "timesteps": idx}

    @torch.no_grad()
    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Here we have a source causing tiny inference-training inconsistency,
        # force convert the state to bf16 then back to float32 to reproduce the info loss in training.
        env_obs["states"] = env_obs["states"].to(torch.bfloat16)
        env_obs["states"] = env_obs["states"].cpu().float()
        opd_raw_states = None
        if self.obs_convert_fn is OBS_CONVERSION["robocasa365"]:
            # Keep the exact (BF16-round-tripped) raw state that the student transform
            # sees.  A frozen suite teacher has different checkpoint statistics and must
            # normalize this state itself during its later NFT pass.
            opd_raw_states = env_obs["states"].detach().cpu().clone()

        observations = self.obs_convert_fn(env_obs)
        # Create a copy to avoid mutating input
        obs_copy = observations.copy()

        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        # Convert to numpy arrays
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        normalized_input = self.apply_transforms(obs_copy)

        if opd_raw_states is not None and not self._opd_state_adapter_validated:
            adapted_state, adapted_mask = self.opd_normalize_robocasa_state(
                opd_raw_states
            )
            torch.testing.assert_close(
                adapted_state,
                normalized_input["state"].float(),
                rtol=0,
                atol=1e-6,
                msg=(
                    "The lightweight OPD state adapter does not reproduce GR00T's full "
                    "student transform."
                ),
            )
            assert torch.equal(adapted_mask, normalized_input["state_mask"].bool())
            self._opd_state_adapter_validated = True

        for key in normalized_input:
            if normalized_input[key].dtype == torch.float32:
                normalized_input[key] = normalized_input[key].to(torch.bfloat16)

        # RoboCasa365: no fixed-length padding. GR00T carries no mask on this path --
        # process_backbone_output runs vlln + vl_self_attention over the whole sequence and
        # the DiT's cross-attention has `encoder_attention_mask` commented out -- so padded
        # positions contaminate the real tokens and are then cross-attended. Measured:
        # padding alone moves the first action dimension from -0.0070 to -0.0746 and takes
        # success_once from 0.88 to 0.00. The processor already pads to the longest prompt
        # in the batch, which is what the reference implementation relies on.
        if self.obs_convert_fn is not OBS_CONVERSION["robocasa365"]:
            normalized_input["eagle_input_ids"] = torch.nn.functional.pad(
                normalized_input["eagle_input_ids"],
                pad=(
                    0,
                    self.padding_value - normalized_input["eagle_input_ids"].shape[-1],
                ),
                mode="constant",
                value=0,
            )
            normalized_input["eagle_attention_mask"] = torch.nn.functional.pad(
                normalized_input["eagle_attention_mask"],
                pad=(
                    0,
                    self.padding_value
                    - normalized_input["eagle_attention_mask"].shape[-1],
                ),
                mode="constant",
                value=0,
            )

        normalized_action, result = self._get_rl_action(normalized_input, mode=mode)
        if opd_raw_states is not None:
            result["forward_inputs"]["opd_raw_states"] = opd_raw_states
        unnormalized_action = self._get_unnormalized_action(normalized_action)

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)

        raw_action = self.action_convert_fn(
            unnormalized_action, chunk_size=self.output_action_chunks
        )

        return torch.from_numpy(raw_action), result

    def apply_transforms(self, obs: dict[str, Any]) -> dict[str, Any]:
        """
        Apply transforms to the observation.

        Args:
            obs (Dict[str, Any]): The observation to transform.

        Returns:
            Dict[str, Any]: The transformed observation.
        """
        # Ensure correct dimensions before applying transforms
        return self._modality_transform(obs)

    def unapply_transforms(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Unapply transforms to the action.

        Args:
            action (Dict[str, Any]): The action to unapply transforms to.

        Returns:
            Dict[str, Any]: The untransformed action.
        """
        return self._modality_transform.unapply(action)

    def _get_rl_action(
        self,
        normalized_input: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ) -> torch.Tensor:
        # We expand get_action() and replace action head inference with RL inference.
        backbone_inputs, action_inputs = self.prepare_input(normalized_input)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs, rlinf_outputs = self.action_head.get_rl_action(
            backbone_outputs, action_inputs, mode=mode
        )
        actions = rlinf_outputs["actions"]
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        actions = actions.float()

        forward_inputs = {
            "chains": rlinf_outputs["chains"],
            "denoise_inds": rlinf_outputs["denoise_inds"],
            **normalized_input,
        }
        bsize = normalized_input["state"].shape[0]
        forward_inputs["eagle_pixel_values"] = normalized_input[
            "eagle_pixel_values"
        ].reshape(
            bsize, self.image_nums, *normalized_input["eagle_pixel_values"].shape[1:]
        )
        forward_inputs["eagle_image_sizes"] = normalized_input[
            "eagle_image_sizes"
        ].reshape(
            bsize, self.image_nums, *normalized_input["eagle_image_sizes"].shape[1:]
        )

        result = {
            "prev_logprobs": rlinf_outputs["prev_logprobs"],
            "prev_values": rlinf_outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }

        return actions, result

    def _get_unnormalized_action(
        self, normalized_action: torch.Tensor
    ) -> dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action.cpu()})

    def opd_normalize_robocasa_state(
        self, raw_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize raw RoboCasa state with this checkpoint's own metadata.

        This is the state-only subset of the full modality pipeline.  Reusing already
        tokenized images and language is safe across these checkpoints, but reusing the
        student's normalized state is not because every suite checkpoint carries its own
        state statistics.
        """
        assert self.obs_convert_fn is OBS_CONVERSION["robocasa365"]
        assert raw_states.shape[-1] == 16, (
            f"RoboCasa OPD expects a 16-d raw state, got {raw_states.shape[-1]}."
        )
        states = raw_states.detach().cpu().float().unsqueeze(-2)
        state_data = {
            key: states[..., start:stop].clone()
            for key, start, stop in ROBOCASA_STATE_SLICES
        }
        state_transform = next(
            (
                transform
                for transform in self._modality_transform.transforms
                if isinstance(transform, StateActionTransform)
                and any(key.startswith("state.") for key in transform.apply_to)
            ),
            None,
        )
        assert state_transform is not None, (
            "GR00T modality pipeline has no state transform."
        )
        state_data = state_transform(state_data)
        state_keys = self._modality_config["state"].modality_keys
        state = torch.cat([state_data[key] for key in state_keys], dim=-1)
        max_state_dim = int(self.config.action_head_cfg["max_state_dim"])
        assert state.shape[-1] <= max_state_dim
        state_mask = torch.zeros(
            *state.shape[:-1], max_state_dim, dtype=torch.bool, device=state.device
        )
        state_mask[..., : state.shape[-1]] = True
        state = torch.nn.functional.pad(
            state, (0, max_state_dim - state.shape[-1]), value=0
        )
        return state, state_mask

    def opd_action_endpoint_affine_to(
        self, target_metadata: DatasetMetadata
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the teacher-endpoint to target-normalized affine action map.

        The returned vectors cover GR00T's padded action dimension.  Continuous RoboCasa
        actions use checkpoint-specific ``min_max`` statistics; binary and padded
        dimensions have an identity map.
        """
        action_dim = int(self.config.action_head_cfg["action_dim"])
        scale = torch.ones(action_dim, dtype=torch.float32)
        shift = torch.zeros(action_dim, dtype=torch.float32)
        action_transform = next(
            (
                transform
                for transform in self._modality_transform.transforms
                if isinstance(transform, StateActionTransform)
                and any(key.startswith("action.") for key in transform.apply_to)
            ),
            None,
        )
        assert action_transform is not None, (
            "GR00T modality pipeline has no action normalization transform."
        )

        offset = 0
        for key in self._modality_config["action"].modality_keys:
            name = key.split(".", 1)[1]
            source_modality = self.metadata.modalities.action[name]
            target_modality = target_metadata.modalities.action[name]
            assert source_modality.shape == target_modality.shape
            dim = int(source_modality.shape[0])
            mode = action_transform.normalization_modes[key]
            if mode == "binary":
                # The released RoboCasa metadata marks these fields as continuous even
                # though the actual training transform is binary.  The transform is the
                # executable contract: both checkpoints map these dimensions to {0, 1},
                # so their endpoint-coordinate map is identity.
                pass
            else:
                assert mode == "min_max", (
                    f"Cross-checkpoint GR00T OPD supports min_max/binary actions, got "
                    f"{mode!r} for {key}."
                )
                source_stats = self.metadata.statistics.action[name]
                target_stats = target_metadata.statistics.action[name]
                source_min = torch.as_tensor(source_stats.min, dtype=torch.float32)
                source_max = torch.as_tensor(source_stats.max, dtype=torch.float32)
                target_min = torch.as_tensor(target_stats.min, dtype=torch.float32)
                target_max = torch.as_tensor(target_stats.max, dtype=torch.float32)
                source_range = source_max - source_min
                target_range = target_max - target_min
                source_degenerate = source_range == 0
                target_degenerate = target_range == 0
                assert torch.equal(source_degenerate, target_degenerate), (
                    f"Source and target disagree on degenerate min_max dimensions for {key}."
                )
                assert torch.all(
                    source_min[source_degenerate] == target_min[target_degenerate]
                ), f"Source and target use different constants for {key}."
                active = ~source_degenerate
                key_scale = torch.ones(dim, dtype=torch.float32)
                key_shift = torch.zeros(dim, dtype=torch.float32)
                key_scale[active] = source_range[active] / target_range[active]
                key_shift[active] = (
                    source_min[active]
                    + source_max[active]
                    - target_min[active]
                    - target_max[active]
                ) / target_range[active]
                scale[offset : offset + dim] = key_scale
                shift[offset : offset + dim] = key_shift
            offset += dim
        assert offset == self.valid_action_dim
        return scale, shift

    def _load_metadata(self, exp_cfg_dir: Path):
        """Load the transforms for the model."""
        # Load metadata for normalization stats
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the metadata.json file is present at {metadata_path}",
            )

        metadata = DatasetMetadata.model_validate(metadata_dict)

        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

        # calculate real intput action dim for rl learning.
        valid_action_dim = 0
        for v in metadata.modalities.action.values():
            valid_action_dim += v.shape[0]
        self.valid_action_dim = valid_action_dim

        self.image_nums = len(metadata.modalities.video.keys())
