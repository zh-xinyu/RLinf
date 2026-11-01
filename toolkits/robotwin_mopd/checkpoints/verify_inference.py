# Copyright 2026 The RLinf Authors.
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
"""Compare fixed-input inference across JAX and converted PyTorch Pi0.5.

Run each backend in a separate process to avoid holding two policies on a GPU.
This validates conversion numerics; it does not measure closed-loop success.
"""

from __future__ import annotations

import argparse
import json
import os
import types
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np


def fixture(tokenizer: Path) -> dict[str, np.ndarray]:
    """Build deterministic images, state, actual Pi0.5 tokens and flow noise."""
    import sentencepiece

    rng = np.random.default_rng(20261101)
    state = rng.uniform(-0.8, 0.8, (1, 32)).astype(np.float32)
    state[:, 14:] = 0
    discrete = np.digitize(state[0], bins=np.linspace(-1, 1, 257)[:-1]) - 1
    prompt = (
        "Task: adjust the bottle, State: "
        + " ".join(map(str, discrete))
        + ";\nAction: "
    )
    encoder = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer))
    tokens = encoder.encode(prompt, add_bos=True)
    assert len(tokens) <= 200
    result = {
        "state": state,
        "tokenized_prompt": np.array(
            [tokens + [0] * (200 - len(tokens))], dtype=np.int32
        ),
        "tokenized_prompt_mask": np.array(
            [[True] * len(tokens) + [False] * (200 - len(tokens))]
        ),
        "noise": rng.standard_normal((1, 50, 32)).astype(np.float32),
    }
    for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        result[key] = rng.uniform(-1, 1, (1, 224, 224, 3)).astype(np.float32)
    return result


def run_jax(args, data):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib
    from openpi.models.pi0 import make_attn_mask
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared.nnx_utils import module_jit

    print("Restoring JAX parameters", flush=True)
    params = model_lib.restore_params(args.checkpoint / "params", dtype=jnp.float32)
    model = Pi0Config(pi05=True, action_horizon=50, dtype="float32").load(params)
    obs = model_lib.Observation(
        images={key: jnp.asarray(data[key]) for key in model_lib.IMAGE_KEYS},
        image_masks={
            key: jnp.ones((1,), dtype=jnp.bool_) for key in model_lib.IMAGE_KEYS
        },
        state=jnp.asarray(data["state"]),
        tokenized_prompt=jnp.asarray(data["tokenized_prompt"]),
        tokenized_prompt_mask=jnp.asarray(data["tokenized_prompt_mask"]),
    )

    def velocity(self, observation, x_t, time):
        prefix, prefix_mask, prefix_ar = self.embed_prefix(observation)
        suffix, suffix_mask, suffix_ar, cond = self.embed_suffix(observation, x_t, time)
        mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        attention = make_attn_mask(mask, jnp.concatenate([prefix_ar, suffix_ar]))
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix, suffix],
            mask=attention,
            positions=jnp.cumsum(mask, axis=1) - 1,
            adarms_cond=[None, cond],
        )
        return self.action_out_proj(suffix_out[:, -50:])

    evaluate_v = module_jit(types.MethodType(velocity, model))
    noise = jnp.asarray(data["noise"])
    outputs = {}
    for i, time in enumerate((1.0, 0.5, 0.1)):
        outputs[f"velocity_{i}"] = np.asarray(evaluate_v(obs, noise, jnp.array([time])))
        print(f"JAX velocity {time} complete", flush=True)
    outputs["actions"] = np.asarray(
        module_jit(model.sample_actions)(
            jax.random.key(7), obs, num_steps=10, noise=noise
        )
    )
    return outputs


def run_torch(args, data):
    import safetensors.torch
    import torch
    from openpi.models import model as model_lib
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

    torch.set_float32_matmul_precision("highest")
    model = PI0Pytorch(Pi0Config(pi05=True, action_horizon=50, dtype="float32"))
    missing, unexpected = safetensors.torch.load_model(
        model, str(args.checkpoint / "model.safetensors"), strict=True
    )
    assert not missing and not unexpected
    model = model.eval().cuda()
    obs = model_lib.Observation(
        images={
            key: torch.from_numpy(data[key]).permute(0, 3, 1, 2).cuda()
            for key in model_lib.IMAGE_KEYS
        },
        image_masks={
            key: torch.ones((1,), dtype=torch.bool, device="cuda")
            for key in model_lib.IMAGE_KEYS
        },
        state=torch.from_numpy(data["state"]).cuda(),
        tokenized_prompt=torch.from_numpy(data["tokenized_prompt"]).cuda(),
        tokenized_prompt_mask=torch.from_numpy(data["tokenized_prompt_mask"]).cuda(),
    )
    with torch.no_grad():
        images, masks, tokens, token_masks, state = model._preprocess_observation(
            obs, train=False
        )
        prefix, pad, ar = model.embed_prefix(images, masks, tokens, token_masks)
        mask = model._prepare_attention_masks_4d(make_att_2d_masks(pad, ar))
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, cache = model.paligemma_with_expert.forward(
            attention_mask=mask,
            position_ids=torch.cumsum(pad, dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=True,
        )
        noise = torch.from_numpy(data["noise"]).cuda()
        outputs = {}
        for i, time in enumerate((1.0, 0.5, 0.1)):
            outputs[f"velocity_{i}"] = (
                model.denoise_step(
                    state, pad, cache, noise, torch.tensor([time], device="cuda")
                )
                .cpu()
                .numpy()
            )
            print(f"PyTorch velocity {time} complete", flush=True)
        outputs["actions"] = (
            model.sample_actions("cuda", obs, noise=noise, num_steps=10).cpu().numpy()
        )
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("jax", "torch", "compare"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    if args.backend == "compare":
        reference, candidate = np.load(args.reference), np.load(args.candidate)
        metrics = {}
        for key in reference.files:
            difference = candidate[key] - reference[key]
            metrics[key] = {
                "max_abs": float(np.max(np.abs(difference))),
                "rmse": float(np.sqrt(np.mean(difference**2))),
                "reference_rms": float(np.sqrt(np.mean(reference[key] ** 2))),
                "relative_rmse": float(
                    np.sqrt(
                        np.mean(difference**2) / (np.mean(reference[key] ** 2) + 1e-12)
                    )
                ),
            }
        args.output.write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))
        # A loose guard to catch conversion errors; inspect actual errors as well.
        assert all(
            np.isfinite(m["relative_rmse"]) and m["relative_rmse"] < 0.01
            for m in metrics.values()
        )
    else:
        data = fixture(args.tokenizer)
        result = run_jax(args, data) if args.backend == "jax" else run_torch(args, data)
        assert all(np.all(np.isfinite(value)) for value in result.values())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, **result)
        print(f"INFERENCE_COMPLETE: {args.output}", flush=True)


if __name__ == "__main__":
    main()
