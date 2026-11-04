# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Check optional online logging before allocating a formal training run."""

import os
from datetime import datetime, timezone


def main() -> None:
    """Write and read a clearly marked setup run, never a paper result."""
    if os.environ.get("WANDB_MODE", "disabled") != "online":
        print("Online logging is disabled; TensorBoard and JSONL remain available.")
        return
    import wandb

    entity = os.environ.get("WANDB_ENTITY")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = wandb.init(
        entity=entity,
        project="FORMAL_robocasa_mopd",
        name=f"setup-logging-{stamp}",
        group="launch_checks",
        job_type="setup",
        tags=["setup", "exclude-from-paper"],
        config={"training_updates": 0},
    )
    run.log({"setup/online_logging_verified": 1})
    path = "/".join(run.path)
    run.finish()
    observed = wandb.Api().run(path)
    if observed.summary.get("setup/online_logging_verified") != 1:
        raise RuntimeError("W&B setup write/read verification failed")
    print(f"Online logging verified: {observed.url}")


if __name__ == "__main__":
    main()
