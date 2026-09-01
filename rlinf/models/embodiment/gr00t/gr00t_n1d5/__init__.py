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

import torch
from omegaconf import DictConfig

from .npu_patches import apply_npu_patches, restore_npu_patches


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from pathlib import Path

    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EmbodimentTag",
        "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
    )
    # RLinf's table follows the N1.6/N1.7 convention and maps "new_embodiment" to 10,
    # while the RoboCasa365 checkpoints were trained against the benchmark fork, which uses
    # 31. That id selects which per-embodiment projector weights the action head uses
    # (state/action encoders and the decoder are all CategorySpecificMLP), so the wrong
    # value silently runs the model on another robot's projectors: no error, correct
    # weights loaded, every input tensor identical -- and useless actions. Leave the
    # installed package's own mapping alone for these tags.
    if cfg.embodiment_tag not in ("new_embodiment", "robocasa_panda_omron"):
        Patcher.add_patch(
            "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
            "rlinf.models.embodiment.gr00t.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        )

    # Register the Ascend-specific patches (no-op off NPU) before applying, and
    # restore the process-global ones once model construction finishes.
    npu_patch_state = apply_npu_patches(Patcher)
    Patcher.apply()
    try:
        # The robocasa-benchmark fork does not ship `load_data_config`, so resolve the
        # "module:ClassName" specs ourselves rather than depending on an API that differs
        # between the two gr00t packages.
        def load_data_config(spec: str):
            import importlib

            module_name, class_name = spec.split(":")
            return getattr(importlib.import_module(module_name), class_name)()


        from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
            GR00T_N1_5_ForRLActionPrediction,
        )
        from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity

        if (
            cfg.embodiment_tag == "libero_franka"
            or cfg.embodiment_tag == "isaaclab_franka"
        ):
            data_config = load_data_config(
                "rlinf.models.embodiment.gr00t.gr00t_n1d5.modality_config:LiberoFrankaDataConfig"
            )
        elif cfg.embodiment_tag == "maniskill_widowx":
            data_config = load_data_config(
                "rlinf.models.embodiment.gr00t.gr00t_n1d5.modality_config:ManiskillWidowXDataConfig"
            )
        elif cfg.embodiment_tag in ("robocasa_panda_omron", "new_embodiment"):
            # Prefer a `panda_omron` config if the installed gr00t registers one -- that is
            # what these checkpoints were trained with. NVIDIA's n1.5-release does not, so
            # fall back to our port, whose fields were checked one by one against it.
            from gr00t.experiment.data_config import DATA_CONFIG_MAP

            if "panda_omron" in DATA_CONFIG_MAP:
                data_config = DATA_CONFIG_MAP["panda_omron"]
            else:
                data_config = load_data_config(
                    "rlinf.models.embodiment.gr00t.gr00t_n1d5.modality_config:RoboCasaPandaOmronDataConfig"
                )
        else:
            raise ValueError(f"Invalid embodiment tag: {cfg.embodiment_tag}")
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        # The transformer registration is done in gr00t/model/gr00t_n1.py
        model_path = Path(cfg.model_path)
        if not model_path.exists():
            # raise error or it triggers auto download from hf(It's cool but we don't have internet connection.)
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        model = GR00T_N1_5_ForRLActionPrediction.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            embodiment_tag=cfg.embodiment_tag,  # This tag determines the state encoder and action head to use
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=cfg.denoising_steps,
            output_action_chunks=cfg.num_action_chunks,
            obs_converter_type=cfg.obs_converter_type,  # TODO(lx): unify the embodiment data format and obs converter
            tune_visual=False,
            tune_llm=False,
            rl_head_config=cfg.rl_head_config,
        )
    finally:
        restore_npu_patches(Patcher, npu_patch_state)

    # `from_pretrained` already applied torch_dtype per module and deliberately keeps the
    # rotary embedding's inverse frequencies in fp32. The blanket `.to()` below overrides
    # that, and those 64 values span several orders of magnitude, so bf16's ~3 significant
    # digits corrupt the positional encoding -- worse at later positions. Snapshot them
    # first and put them back afterwards: casting bf16 back to fp32 would *not* recover the
    # lost precision (0.80584 -> 0.8046875 is permanent).
    preserved_inv_freq = {
        name: module.inv_freq.detach().clone()
        for name, module in model.named_modules()
        if isinstance(getattr(module, "inv_freq", None), torch.Tensor)
        and module.inv_freq.is_floating_point()
    }

    model.to(torch_dtype)

    modules_by_name = dict(model.named_modules())
    for name, original in preserved_inv_freq.items():
        modules_by_name[name].inv_freq = original

    if cfg.rl_head_config.add_value_head:
        # reinitialize the value head after model loading, or there are nan values in the value head after model loading.
        model.action_head.value_head._init_weights()

    if cfg.rl_head_config.disable_dropout:
        replace_dropout_with_identity(model)

    return model
