# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Export a saved offline head as a standard HF student, without further training."""

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from toolkits.libero_gr00t.offline_observations import file_hash
from toolkits.libero_gr00t.probe import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if not (args.checkpoint / "COMPLETE").exists():
        raise ValueError("Require a complete offline checkpoint")
    receipt = json.loads((args.checkpoint / "manifest.json").read_text())
    head = args.checkpoint / "action_head.safetensors"
    if file_hash(head) != receipt["head_sha256"]:
        raise ValueError("Saved action head changed")
    destination = args.checkpoint / "hf"
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(1)
    base = Path(receipt["base_checkpoint"])
    cfg = SimpleNamespace(
        student=base,
        execute=10,
        denoise_steps=4,
        noise_level=0.3,
        student_head_precision="fp32",
        student_text_padding="native",
    )
    model = load_model(base, cfg).cpu().eval().requires_grad_(False)
    tensors = load_file(str(head), device="cpu")
    expected = {
        k: v for k, v in model.state_dict().items() if k.startswith("action_head.")
    }
    if set(tensors) != set(expected):
        raise ValueError("Require complete action-head keys")
    for name, tensor in tensors.items():
        if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
            raise ValueError(f"Action-head shape/dtype mismatch: {name}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite tensor: {name}")
    model.load_state_dict(tensors, strict=False)
    model.save_pretrained(destination, safe_serialization=True, max_shard_size="5GB")
    shutil.copytree(base / "experiment_cfg", destination / "experiment_cfg")
    (destination / "offline_export.json").write_text(
        json.dumps(
            {
                "optimizer_update": receipt["optimizer_updates"],
                "source_head_sha256": receipt["head_sha256"],
                "metadata_sha256": file_hash(
                    destination / "experiment_cfg/metadata.json"
                ),
                "additional_optimizer_updates": 0,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
