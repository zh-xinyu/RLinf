# The gr00t package — how it is wired in, and why the fork was dropped

Verified on node `action00`, 2026-08-22; re-verified without the fork 2026-08-23.

> **Setup commands live in [RUNBOOK.md](RUNBOOK.md) §2.** They are not repeated here, so the
> two cannot drift apart. This page keeps what the RUNBOOK does not: how the gr00t package is
> wired in, how the two candidate packages compared, and how much a 16-env number moves on its
> own.

## How the gr00t package is wired in

`install.sh` installs it with `uv pip install -e $VENV/gr00t`, which writes a *pointer* (an
editable finder), not a copy:

```
site-packages/__editable__.gr00t-1.1.0.pth
  -> __editable___gr00t_1_1_0_finder.py
  -> {'gr00t': '$VENV/gr00t/gr00t'}
```

**This lives in the venv, not in git**, so it must be redone on every new node — and it is
also why swapping gr00t implementations during the investigation never touched a source file
or any git state.

## How different the two packages actually were

Kept because it corrects a claim that misdirected the original investigation. Measured with
`diff` on the inference path:

| file | diff lines | what it is |
|---|---|---|
| `model/gr00t_n1.py` | 0 | identical |
| `model/transforms.py` | 0 | identical |
| `data/embodiment_tags.py` | 0 | identical — both map `new_embodiment` to 31 |
| `action_head/flow_matching_action_head.py` | 7 | import line wrapping only |
| `eagle2_5_vl` processor | 75 | `decord` video-reading helpers; not on the image path we use |
| `data/transform/state_action.py` | 169 | the fork vendors pytorch3d's quaternion helpers to drop the dependency |
| `experiment/data_config.py` | 400 | **the fork registered `panda_omron`; NVIDIA's does not** |

Near-identical where it counts. The fork carried the RoboCasa data config; NVIDIA's package
was never broken. **RESULTS.md originally listed the package mismatch as one of the defects —
that was wrong**, and it cost a round of debugging.

The one RLinf-side consequence: `load_data_config` does not exist in the fork, so
`gr00t_n1d5/__init__.py` resolves `"module:ClassName"` specs locally. The inheritance
(`GR00T_N1_5_ForRLActionPrediction`, the RL action head) needed no change at all, which
follows from `gr00t_n1.py` and `transforms.py` being byte-identical.

## How much does a 16-env number move on its own?

Worth knowing before reading any single result. The atomic teacher, official package, same
tasks, same horizon, 16 envs — four independent runs:

| run | path | success_once | success_at_end |
|---|---|---|---|
| first official-package test | delegation | 0.75 | 0.6875 |
| `eval_all_ckpts` | delegation | 0.8125 | 0.75 |
| `ab_rlinf_path` | RLinf `_get_rl_action` | 0.875 | 0.8125 |
| `post_del_atomic` (delegation deleted) | RLinf `_get_rl_action` | 0.875 | 0.8125 |
| `final_nofork` (fork deleted) | RLinf `_get_rl_action` | 0.9375 | 0.875 |

**The same configuration spans 0.75–0.9375 — three episodes — purely from the initial noise
draw.** So a two-episode gap between any two conditions here means nothing, and even the
five-episode gap seen once between the two inference paths was shown by direct tensor
comparison to be sampling, not code (see RESULTS.md). Comparisons at n=16 resolve only large
effects, such as the composite teacher-vs-student gaps of 7 and 11 episodes.

## The fork was dropped (2026-08-23)

Its only contribution was the `panda_omron` data config. Our `RoboCasaPandaOmronDataConfig`
was checked against it field by field — **16/16 identical**: video / state / action /
language keys, observation and action indices, all six modality-config delta-index and key
lists, and the transform pipeline's type order — with the fork loaded from its source tree
and our port instantiated under the official package. All four released checkpoints then
evaluated end to end on the official package, every run `status=OK`.

Keeping only NVIDIA's package is deliberate: the fork is a third-party repo that could
disappear, which would make the environment unreproducible. The loader still prefers
`panda_omron` if some future gr00t provides it, so a fork install would change nothing that
matters — but nothing requires one.

**The residual obligation:** we now maintain that data config ourselves. Inferring it wrongly
is exactly what started the original zero-success investigation, so if these checkpoints are
ever replaced, re-run the field-by-field comparison rather than assuming it still holds.
