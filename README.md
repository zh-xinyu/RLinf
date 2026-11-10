# Multi-Teacher On-Policy Distillation for Robot Policies

Built on [RLinf](https://github.com/RLinf/RLinf) at upstream commit
[`e7609d4c`](https://github.com/RLinf/RLinf/commit/e7609d4c9e2f33c5ffc10b67c61c8e4b73208b45).

This LIBERO experiment branch consolidates four task-specialist GR00T N1.5
policies into one student across Spatial, Object, Goal and Long. We study
adaptive rollout allocation with synchronized suite loss weights and decoupled
flow refresh, which reuses physical observations while refreshing the current
student's flow trajectories and teacher supervision.

| Purpose | Entry |
| --- | --- |
| Check current setting and acceptance status | [Experiment contract](LIBERO_EXPERIMENTS.md) |
| Request a cluster image and install manually | [Environment guide](LIBERO_ENVIRONMENT.md) |
| Inspect shared rollout-worker evaluation | [Evaluation protocol](toolkits/libero_gr00t/parallel_evaluation.md) |
| Inspect implementation and historical pilots | [Toolkit notes](toolkits/libero_gr00t/README.md) |
| Reproduce the separate RoboCasa environment | [RoboCasa profile](OPD_ENVIRONMENT.md) |

The current LIBERO source is a freeze candidate. The new teacher-rollout SFT
initializer still requires complete screening, confirmation and a short MOPD
transition check. New cluster hardware also requires admission tests. The
experiment contract lists these gates explicitly; do not treat historical
Offline-Distilled-50 experiments as results of the new initialization.

The default main matrix is four arms, one node/four GPUs, T64trajectories/round,
100rounds, B512 and Refresh-R1/R2. Independent evaluation uses40tasks ×50states
per saved checkpoint through the training environment/rollout implementation.
Installation is isolated from RoboCasa and requires no personal node profiles.

Checkpoints, raw experiments, credentials and machine paths are distributed
separately. Source/configuration files and portable setup instructions belong in
Git. The inherited RLinf Sphinx docs and Chinese README retain upstream/RoboCasa
context; the English LIBERO entries above are authoritative for this branch.
