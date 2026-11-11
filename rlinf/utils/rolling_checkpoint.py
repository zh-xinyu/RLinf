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
"""Keep the newest complete FSDP recovery state and every policy weight file."""

import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

from rlinf.utils.logging import get_logger

logger = get_logger()


def _atomic_json(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
            os.fchmod(stream.fileno(), mode)
            json.dump(data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def publish_rolling_checkpoint(
    checkpoint: str | Path, *, require_allocation: bool = False
) -> list[Path]:
    """Publish a validated recovery point and prune older managed DCP states.

    Policy weights and rollout/runner JSON files are retained at every save point.
    Only older checkpoints explicitly marked as rolling-managed can lose their
    DCP state. Validation failure leaves the previous recovery point intact.

    Args:
        checkpoint: Newly completed ``global_step_<N>`` checkpoint directory.
        require_allocation: Require matching rollout-allocation metadata.

    Returns:
        Older DCP directories removed after the new recovery point was published.

    Raises:
        ValueError: The new checkpoint is incomplete or has inconsistent counters.
    """
    from torch.distributed.checkpoint import FileSystemReader

    checkpoint = Path(checkpoint)
    match = re.fullmatch(r"global_step_(\d+)", checkpoint.name)
    state = json.loads((checkpoint / "runner_state.json").read_text())
    if (
        match is None
        or state.get("weights_only", True)
        or not state.get("rolling_recovery_managed", False)
        or state.get("step_unit") != "optimizer_update"
        or state.get("optimizer_updates") != int(match.group(1))
    ):
        raise ValueError("Incomplete or inconsistent rolling recovery checkpoint.")
    weights = checkpoint / "actor/model_state_dict/full_weights.pt"
    if not zipfile.is_zipfile(weights):
        raise ValueError("Rolling recovery policy weights are incomplete.")
    if require_allocation:
        allocation = json.loads(
            (checkpoint / "rollout_allocation_state.json").read_text()
        )
        if (
            allocation["rounds"] != state["rollout_count"]
            or allocation["simulation_steps"] != state["env_steps"]
        ):
            raise ValueError("Rolling recovery allocation counters do not match.")
    dcp = checkpoint / "actor/dcp_checkpoint"
    metadata = FileSystemReader(str(dcp)).read_metadata()
    keys = metadata.state_dict_metadata
    for component in ("model", "optimizers", "lr_schedulers", "rng"):
        if not any(key.startswith(f"fsdp_checkpoint.{component}.") for key in keys):
            raise ValueError(f"Rolling recovery is missing {component} state.")
    extents = {}
    for item in metadata.storage_data.values():
        relative = Path(item.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unexpected DCP shard path.")
        extents[relative] = max(extents.get(relative, 0), item.offset + item.length)
    if not extents or any(
        not (dcp / path).is_file() or (dcp / path).stat().st_size < size
        for path, size in extents.items()
    ):
        raise ValueError("Rolling recovery DCP shard is missing or truncated.")

    _atomic_json(
        checkpoint.parent / "latest_full_checkpoint.json",
        {
            "version": 1,
            "checkpoint": checkpoint.name,
            "optimizer_updates": state["optimizer_updates"],
        },
    )
    removed = []
    for older in checkpoint.parent.glob("global_step_*"):
        old_match = re.fullmatch(r"global_step_(\d+)", older.name)
        if (
            old_match is None
            or int(old_match.group(1)) >= state["optimizer_updates"]
            or older.is_symlink()
            or older.stat().st_uid != os.getuid()
        ):
            continue
        old_state_path = older / "runner_state.json"
        if not old_state_path.is_file():
            continue
        try:
            old_state = json.loads(old_state_path.read_text())
        except (OSError, ValueError):
            logger.warning("Preserving unreadable older checkpoint %s", older)
            continue
        if not old_state.get("rolling_recovery_managed", False) or old_state.get(
            "optimizer_updates"
        ) != int(old_match.group(1)):
            continue
        old_dcp = older / "actor/dcp_checkpoint"
        if (older / "actor").is_symlink() or old_dcp.is_symlink():
            continue
        if not old_dcp.is_dir():
            continue
        # Mark first: a crash during pruning cannot advertise a missing full state.
        old_state.update(
            weights_only=True, recovery_state_superseded_by=checkpoint.name
        )
        _atomic_json(old_state_path, old_state)
        try:
            shutil.rmtree(old_dcp)
        except OSError:
            logger.warning("Could not prune older recovery state %s", old_dcp)
        else:
            removed.append(old_dcp)
    return removed
