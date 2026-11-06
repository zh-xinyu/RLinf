# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tiny four-GPU NCCL/FSDP/Adam/save check, not a full VLA training smoke."""

import argparse
import copy
import ctypes
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy


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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        losses = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(torch.randn(8, 64, device="cuda")).float().square().mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
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
        # Exercise metadata collectives too: small all-reduces alone missed a
        # Blackwell NCCL failure in the full model's DCP save path.
        payload = [bytes([i]) * (4 * 1024**2) for i in range(4)] if rank == 0 else None
        received = [None]
        dist.scatter_object_list(received, payload, src=0)
        assert received[0] == bytes([rank]) * (4 * 1024**2)
        expected_parameters = [p.detach().clone() for p in model.parameters()]
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_scheduler = copy.deepcopy(scheduler.state_dict())
        checkpoint = str(args.output / "production_checkpoint")
        FSDPStrategy.save_checkpoint(model, optimizer, scheduler, checkpoint)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1)
            for state in optimizer.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        value.zero_()
        scheduler.step()
        FSDPStrategy.load_checkpoint(model, optimizer, scheduler, checkpoint)
        for actual, expected in zip(model.parameters(), expected_parameters):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            optimizer.state_dict(), expected_optimizer, rtol=0, atol=0
        )
        assert scheduler.state_dict() == expected_scheduler
        runtime_nccl = ctypes.c_int()
        assert (
            ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(runtime_nccl)) == 0
        )
        report = {
            "rank": rank,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl_runtime_version": runtime_nccl.value,
            "ld_preload": os.environ.get("LD_PRELOAD", ""),
            "losses": losses,
            "max_allocated_bytes": torch.cuda.max_memory_allocated(rank),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(rank),
            "status": "PASS",
            "scope": "tiny FSDP, 4MiB scatter, production DCP/full-weight save and exact model/Adam/scheduler reload; not full VLA coverage",
        }
        (args.output / f"rank{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
