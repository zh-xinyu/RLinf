# Multi-Teacher On-Policy Distillation for Robot Policies

Built on [RLinf](https://github.com/RLinf/RLinf) at upstream commit [`e7609d4c`](https://github.com/RLinf/RLinf/commit/e7609d4c9e2f33c5ffjoint-sft-fullb67c61c8e4b73208b45).

This project consolidates task-specialist vision-language-action policies into
one flow-based student. Our current setting uses GR00T N1.5 and RoboCasa365's
atomic-seen, composite-seen, and composite-unseen suites: 50 manipulation tasks,
three specialist teachers, and a joint student initialized from full-data joint SFT
(Joint-SFT-Full: 500 demonstrations per task, 25,000 demonstrations total).

We study two complementary mechanisms:

- **Adaptive rollout allocation and synchronized suite loss weights:** direct
  physical experience and optimization toward remaining teacher–student gaps.
- **Decoupled flow refresh:** reuse a physical rollout's observation pool while
  regenerating flow trajectories and teacher targets with the updated student.

## Start here

| Purpose | Entry |
| --- | --- |
| Reproduce the current software environment | [Fixed GR00T/RoboCasa365 profile](OPD_ENVIRONMENT.md) |
| Run the four agreed experiments, including setup and evaluation | [Collaborator / coding-agent handoff](docs/opd/collaboration/README.md) |
| Obtain and verify Joint-SFT-Full, teachers, and simulator assets | [Resources](docs/opd/collaboration/RESOURCES.md) |
| Understand the training implementation and metrics | [Development guide](OPD_DEVELOPMENT.md) |
| Inspect the independent evaluation protocol | [RoboCasa365 evaluation](evaluations/robocasa365/FORMAL_EVAL.md) |

The experiment release is pinned by Git tag `robocasa-mopd-joint-sft-full-v1`.
Use that tag for reproduction; `opd-project-main` remains the development base.
Training uses eight 48-GB GPUs on one machine. Every arm starts from Joint-SFT-Full,
collects 96 trajectories per rollout round, runs 150 rounds, and saves every
25 rounds. The handoff defines all remaining controls and result requirements.

Checkpoints and simulator assets are distributed separately. Personal node
paths, credentials, raw runs, and working research notes are not installation
dependencies. RoboTwin experiments will use a separate branch and environment
profile; the current profile only covers GR00T N1.5 + RoboCasa365.

Project-facing guides are maintained in English; a [Chinese README](README.zh-CN.md)
is also available. Email task assignments and local research notes are distributed
separately. The inherited RLinf documentation under `docs/` has not yet been
fully adapted to this project; use the OPD entries above for reproduction.

## Reproduce an experiment

The four training YAMLs are already provided. Copy
[`machine.example.json`](toolkits/robocasa_mopd/machine.example.json) outside the
repository and fill in your hostname and paths once. W&B is optional and disabled
by default; local TensorBoard/JSONL logs and the summary CSV are always required.
Every stage uses the same entrypoint; no shell exports or manual runtime activation
are needed:

```bash
python3 toolkits/robocasa_mopd/experiment.py --machine /path/to/machine.json paths
python3 toolkits/robocasa_mopd/experiment.py --machine /path/to/machine.json install
# Complete the resource and smoke steps in the experiment guide before training.
python3 toolkits/robocasa_mopd/experiment.py --machine /path/to/machine.json train m0
python3 toolkits/robocasa_mopd/experiment.py --machine /path/to/machine.json eval m0
python3 toolkits/robocasa_mopd/experiment.py --machine /path/to/machine.json summarize
```

## Repository map

- `rlinf/`: distributed training, simulation, model adapters, and distillation.
- `examples/embodiment/config/robocasa365_mopd_formal_*.yaml`: frozen matrix.
- `toolkits/robocasa_mopd/`: portable resource checks and experiment launchers.
- `evaluations/robocasa365/`: task manifest and independent evaluation.
- `requirements/profiles/`: versioned software environments.
- `tests/unit_tests/`: protocol, loss, allocation, refresh, and memory regressions.

## Attribution and license

The repository retains RLinf's [Apache-2.0 license](LICENSE) and upstream
copyright notices. External model checkpoints, simulator assets, and third-party
software retain their respective licenses. See [CONTRIBUTING.md](CONTRIBUTING.md)
for code conventions inherited from RLinf.
