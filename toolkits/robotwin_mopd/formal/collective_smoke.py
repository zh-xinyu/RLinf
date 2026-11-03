# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tiny four-GPU NCCL/FSDP/Adam/save check, not a full VLA training smoke."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(0.05, rank)
    dist.init_process_group("nccl")
    try:
        if dist.get_world_size() != 4:
            raise ValueError("This deployment test requires four processes")
        torch.manual_seed(123)
        model = FSDP(
            torch.nn.Sequential(
                torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 64)
            ).cuda(),
            device_id=rank,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        losses = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(torch.randn(8, 64, device="cuda")).float().square().mean()
            loss.backward()
            optimizer.step()
            metric = loss.detach().clone()
            dist.all_reduce(metric)
            if not torch.isfinite(metric):
                raise ValueError("Nonfinite distributed loss")
            losses.append(metric.item() / dist.get_world_size())
        # A small per-rank checkpoint checks optimizer and parameter serialization.
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        torch.save(
            {
                "parameters": [p.detach().cpu() for p in model.parameters()],
                "optimizer": optimizer.state_dict(),
            },
            args.output / f"rank{rank}.pt",
        )
        report = {
            "rank": rank,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "losses": losses,
            "max_allocated_bytes": torch.cuda.max_memory_allocated(rank),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(rank),
            "status": "PASS",
            "scope": "tiny FSDP test; not VLA model/runner checkpoint coverage",
        }
        (args.output / f"rank{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
