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
"""Convert a pinned Pi0.5 Orbax checkpoint with audited weight loading."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument(
        "--precision", default="float32", choices=("float32", "bfloat16")
    )
    args = parser.parse_args()
    import torch
    from openpi.models.pi0_config import Pi0Config

    converter_path = args.openpi_root / "examples/convert_jax_model_to_pytorch.py"
    spec = importlib.util.spec_from_file_location("upstream_converter", converter_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    loads = []
    original_load = torch.nn.Module.load_state_dict

    def audited_load(model, state_dict, *load_args, **load_kwargs):
        result = original_load(model, state_dict, *load_args, **load_kwargs)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        # The action-only Gemma expert does not use its lm_head. PaliGemma's
        # output head must share storage with the supplied input embedding.
        permitted = {"paligemma_with_expert.gemma_expert.lm_head.weight"}
        embedding_key = (
            "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        )
        head_key = "paligemma_with_expert.paligemma.lm_head.weight"
        parameters = dict(model.named_parameters(remove_duplicate=False))
        if (
            embedding_key in state_dict
            and parameters[embedding_key].data_ptr() == parameters[head_key].data_ptr()
        ):
            permitted.add(head_key)
        bad_missing = [key for key in missing if key not in permitted]
        if bad_missing or unexpected:
            raise ValueError(
                f"Incomplete conversion: missing={bad_missing}; unexpected={unexpected}"
            )
        loads.append(
            {
                "loaded_keys": len(state_dict),
                "missing_keys": missing,
                "unexpected_keys": unexpected,
            }
        )
        return result

    torch.nn.Module.load_state_dict = audited_load
    try:
        module.convert_pi0_checkpoint(
            str(args.source),
            args.precision,
            str(args.destination),
            Pi0Config(pi05=True, action_horizon=50, dtype="float32"),
        )
    finally:
        torch.nn.Module.load_state_dict = original_load
    # The upstream converter looks one directory too high for checkpoint-local
    # assets. Explicitly preserve this exact checkpoint's original statistics.
    if (args.source / "assets").exists():
        shutil.copytree(
            args.source / "assets", args.destination / "assets", dirs_exist_ok=True
        )
    config = json.loads((args.destination / "config.json").read_text())
    config.update({"pi05": True, "discrete_state_input": True, "max_token_len": 200})
    (args.destination / "config.json").write_text(json.dumps(config, indent=2))
    weight = args.destination / "model.safetensors"
    with weight.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = {
        "source": str(args.source),
        "destination": str(args.destination),
        "openpi_revision": subprocess.check_output(
            ["git", "-C", str(args.openpi_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "converter_sha256": hashlib.sha256(converter_path.read_bytes()).hexdigest(),
        "precision": args.precision,
        "action_horizon": 50,
        "weight_sha256": digest,
        "weight_bytes": weight.stat().st_size,
        "load_audit": loads,
        "validation": "parameter_mapping_only; closed-loop and JAX/PyTorch equivalence pending",
    }
    (args.destination / "conversion.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
