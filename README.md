# Multi-Teacher On-Policy Distillation for Robot Policies

Built on [RLinf](https://github.com/RLinf/RLinf) at upstream commit
[`e7609d4c`](https://github.com/RLinf/RLinf/commit/e7609d4c9e2f33c5ffc10b67c61c8e4b73208b45).

This LIBERO experiment branch consolidates four task-specialist GR00T N1.5
policies into one student across Spatial, Object, Goal and Long. We study
adaptive rollout allocation with synchronized query counts and decoupled
flow refresh, which reuses physical observations while refreshing the current
student's flow trajectories and teacher supervision.

For collaborators, start with **[LIBERO_HANDOFF.md](LIBERO_HANDOFF.md)**. It contains
the assigned matrix, machine configuration, launch commands and required results.
The installation guide is needed once; the other pages below are references.

| Purpose | Entry |
| --- | --- |
| Run the current six Offline50 experiments | [80-round commands and results](LIBERO_OFFLINE50.md) |
| Update an existing installation | [Experiment handoff](LIBERO_HANDOFF.md) |
| Install on a new approved platform image | [Cluster setup](LIBERO_CLUSTER_SETUP.md) |
| Inspect reference environment profiles | [Environment guide](LIBERO_ENVIRONMENT.md) |
| Inspect shared rollout-worker evaluation | [Evaluation protocol](toolkits/libero_gr00t/parallel_evaluation.md) |
| Inspect the adaptive law | [Adaptive query protocol](toolkits/libero_gr00t/adaptive_query.md) |
| Inspect earlier 100-round experiments | [Historical contract](LIBERO_EXPERIMENTS.md) |
| Reproduce the separate RoboCasa environment | [RoboCasa profile](OPD_ENVIRONMENT.md) |

The current matrix is Uniform-R1/R2 and Adaptive-I5/I10-R1/R2, seed0,
80physical rollout rounds, T64 and B512. All methods use the restored
Offline-Distilled-50 initialization and unchanged specialist teachers. The two
Adaptive variants keep the squared-gap law and differ only in decision interval;
allocator warmup/window are10rounds. The additional SFT100 export and older
100-round controls remain separate historical artifacts.

New `queue_offline50` / `run_offline50` commands use the existing runtime and
machine JSON, preserve local resource/logging choices, and automatically train,
evaluate and export results. Three four-GPU servers can run the six jobs with
one two-job queue per server. Intermediate curves use20episodes/task; the shared
initial and final results use50episodes/task through the training rollout path.
New output/CSV namespaces prevent mixing with previous experiments. No dependency
profile or environment rebuild is required for this update. See the current
command guide for CPU validation evidence and the cloud acceptance boundary.

Checkpoints, raw experiments, credentials and machine paths are distributed
separately. Source/configuration files and portable setup instructions belong in
Git. The inherited RLinf Sphinx docs and Chinese README retain upstream/RoboCasa
context; the English LIBERO entries above are authoritative for this branch.
