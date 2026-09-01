# Routed MOPD — results

Setting, checkpoints and the working recipe: [README.md](README.md).

> **Correction (2026-08-31):** the per-suite M4 `opd_reward` analysis later in this file is
> invalid. It grouped the full rollout even though only suite-balanced sampled rows were
> scored, so unscored zero-filled rows diluted the suite statistics. The old suite means,
> signal-to-noise estimate, and correlation claim are withdrawn. Behavioral success metrics
> remain historical observations. The implementation now reports only the sampled rows;
> existing logs cannot be used to reconstruct the corrected diagnostic.

## Pilot configuration

| | |
|---|---|
| student init | GR00T N1.5 `foundation_model_learning/pretraining/checkpoint-80000` |
| teachers | `target_posttraining/{atomic_seen,composite_seen,composite_unseen}/checkpoint-60000` |
| routing | task suite -> teacher, deterministic |
| tasks | 2 per suite: TurnOnMicrowave, CloseToasterOvenDoor / ScrubCuttingBoard, RinseSinkBasin / CuttingToolSelection, CategorizeCondiments |
| split | `target` |
| train | 64 envs, episodes truncated at 300 steps, lr 2e-5, batch 3840, 10 steps |
| eval | 48 envs, 750-step cap, fixed reset ids, ordered task sampling |
| primary metric | `success_once` |

## Training signal

(filled in as the run proceeds)

| step | velocity_mse | velocity_cosine |
|---|---|---|
| 1 | 0.0030 | 0.768 |
| 2 | 0.0016 | 0.888 |
| 3 | 0.0015 | |
| 4 | 0.0012 | |
| 5 | 0.0010 | |
| 6 | 9.43e-04 | |
| 7 | 9.51e-04 | |
| 8 | 9.10e-04 | |
| 9 | 9.06e-04 | |
| 10 | 8.55e-04 | |

The loss falls 3.5x over ten steps and then plateaus around 9e-04 from step 6 on: the
student has absorbed most of what a routed velocity match can give it at this budget, and
further steps at this learning rate would buy little. **Validation at both step 5 and step
10 is `success_once = 0.0` / `success_at_end = 0.0`.**

The student is moving toward its routed teachers. Cosine starting at 0.77 rather than ~1
confirms the specialists differ substantially from the pretrained student, so there is a
real gradient to follow.

## Success rates

(pending: in-training validation at steps 5 and 10, plus a standalone baseline evaluation
of the undistilled student through `evaluations/robocasa365/robocasa365_mopd_gr00t_eval.yaml`)

## Caveats to carry into any writeup

- One seed, one configuration, six tasks. Nothing here supports a claim about the full
  50-task leaderboard.
- Training episodes are truncated at 300 steps, so the on-policy state distribution only
  covers the early part of each episode.
- The composite suites have horizons of 1200-1650 steps; at this pilot's budget they are
  expected to score ~0, so essentially all of the success signal comes from the two atomic
  tasks. The routing machinery is nevertheless exercised by all three suites.
- No single-teacher baseline and no teacher-disagreement diagnostic were run, so this
  establishes "routed multi-teacher OPD trains", not "routing helps".

## First validation: zero success at step 5

In-training validation after 5 distillation steps: `success_once=0.0`, `success_at_end=0.0`
across the six pilot tasks (48 envs, 750-step cap). Training-side distillation is clearly
working (`velocity_mse` 0.0030 -> 0.0010 over five steps, cosine 0.77 -> ~0.9), so this is
about the policy's absolute competence, not the loss.

Three candidate explanations, in the order they should be ruled out:

1. **Our GR00T <-> RoboCasa interface is wrong.** The modality config was reconstructed from
   the checkpoint's `metadata.json`; the action concat order
   (`[eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)]`) and the
   `min_max` normalization modes were *inferred*, not read from the RoboCasa365 training
   code, which is not in either installed repo.
2. **The undistilled foundation checkpoint is simply near zero on these tasks**, which would
   make 0 the honest baseline rather than a bug.
3. **Five steps is too little.**

**The decisive test is (1), and it is queued:** evaluate the `atomic_seen` *teacher* on the
two atomic tasks through the identical harness. That teacher was post-trained on exactly
those tasks, so if it also scores 0 the interface is wrong; if it scores well the interface
is right and the question becomes (2) vs (3). Two eval jobs are armed and chained on status
files: `eval.baseline` (undistilled student, all six tasks) then `eval.teacher_atomic`.

Do not read anything into the pilot's success numbers until that diagnostic returns.

## The teacher diagnostic: it is the interface, and it is not yet fixed

| model | tasks | success_once |
|---|---|---|
| student, 10 distillation steps | 6 pilot tasks | 0.0 |
| student, undistilled (baseline) | 6 pilot tasks | 0.0 |
| **`atomic_seen` teacher** | the 2 atomic tasks it was post-trained on | **0.0 (0/16)** |
| `atomic_seen` teacher, after the data-config fix below | same 2 tasks | **0.0 (0/16)** |

A specialist scoring 0/16 on its own tasks, with a 750-step cap against 450-step horizons,
cannot be explained by task difficulty or by the distillation. The GR00T <-> RoboCasa
interface is wrong.

### Confirmed and fixed: the modality config was reconstructed wrongly

Isaac-GR00T n1.5-release ships the authoritative config as
`gr00t.experiment.data_config.SinglePandaGripperDataConfig` (same key names, same robot).
Ours differed in four ways, each of which silently changes what the model is fed:

| | ours (inferred) | authoritative |
|---|---|---|
| state concat order | base_pos, base_rot, eef_pos, eef_rot, gripper | **eef_pos, eef_rot, gripper, base_pos, base_rot** |
| rotations | raw quaternions | **`rotation_6d` for both quaternion state keys** (state is 20-d, not 16-d) |
| `gripper_close`, `control_mode` | `min_max` | **`binary`** |
| video order | left, wrist, right | upstream is left, right, wrist |

Ported verbatim. Independently confirmed *correct* in our original port, against
`robocasa/utils/lerobot_utils.py`: the action layout
(`eef_pos 0:3, eef_rot 3:6, gripper 6:7, base_motion 7:11, control_mode 11:12`) and the
state key -> hdf5 key mapping. Actions take normalization but no rotation target, so the
emitted action is 12-d as implemented.

### Ruled out

- **Success plumbing.** `rlinf/envs/robocasa/venv.py` explicitly calls `env._check_success()`
  and writes `info["success"]`, which the env reads. Zero is a real zero, not a missing key.
- **Image orientation.** `flip_images_vertical` defaults to True and is applied to all three
  cameras.
- **Prompt truncation.** The initial 570 -> 1024 change removed truncation; the durable fix
  later disabled fixed-length padding on RoboCasa365 because padded, unmasked tokens changed
  the policy output.

### Next test, and why

Run the **shipped** `evaluations/robocasa365/robocasa365_eval_openpi.yaml` with an openpi
RoboCasa checkpoint. That path is upstream code we did not write. If it also scores 0, the
problem is in the environment or the evaluation protocol (reset-state distribution, seeding,
horizon convention) rather than in the GR00T port, and the whole search moves. This is the
mechanism study's "reproduce it with code you didn't write" lesson.

Then, in order: the action application path
(`prepare_actions_for_robocasa` + `binarize_gripper_control` + `base_mode_index`, which may
now double-transform the gripper given `binary` normalization), and the range convention of
`unapply_transforms` output versus what robosuite expects.

## The reference run: the checkpoint is fine, our port is not

The user had already validated these exact checkpoints on RoboCasa365 in
`platform_feasibility_probe/robocasa365_variant_a/`. From
`logs/probe_student_n20_213630.log`:

```
[probe] label=student_n20 task=CloseFridge split=pretrain n_episodes=20 n_envs=20 n_action_steps=16
EP 19 success: True; Cumulative success rate: 0.8947
```

That "student" is `pretraining/checkpoint-80000` -- **the same undistilled checkpoint we
measured at 0.0**. It reaches ~0.9 on CloseFridge. So the checkpoint is competent, the task
is achievable, and the RLinf-side port is what is broken. This retires every remaining
"maybe it's just hard" hypothesis.

### What the reference setup does differently

| | ours | official (validated) |
|---|---|---|
| executed action steps | `num_action_chunks: 5` | **`n_action_steps: 16`** (the model's full action horizon) |
| episode horizon | capped at 750 | `get_task_horizon(task)`, per task |
| camera order | eye_in_hand first (from metadata.json) | **agentview_left, agentview_right, eye_in_hand** at 256x256 |
| eval path | RLinf env + our converters | GR00T `SimulationInferenceClient` + `robocasa/wrappers/gym_wrapper.py` |
| data config | our port of `SinglePandaGripperDataConfig` | `panda_omron` (a fork registration; the fork itself is gone, but `single_panda_gripper` in stock n1.5 has the same robot and keys) |

`robocasa/wrappers/gym_wrapper.py` is installed in our venv and *is* the official
observation/action contract: it maps `body.*`/`hand.*` to `state.*` (so our state key names
were right), declares the camera list and 256x256 resolution, and defines the action dict.

### Fixed since

- Camera order corrected to the wrapper's (`agentview_left, agentview_right, eye_in_hand`).

### Still to do, in order

1. **`num_action_chunks` 5 -> 16.** The reference executes the full 16-step action horizon;
   emitting only the first 5 changes the control regime entirely and is the largest
   remaining discrepancy.
2. **Re-run the teacher diagnostic** after (1). The bar is no longer "non-zero" but
   "comparable to ~0.9 on an easy atomic task".
3. **Port the action application from `gym_wrapper.py` rather than inferring it** --
   specifically the gripper and control-mode thresholds (`-1.0 if x < 0.5 else 1.0`) versus
   RLinf's `binarize_gripper_control` / `base_mode_index`, which may now double-transform
   given the data config normalizes those two keys as `binary`.
4. Use per-task horizons for evaluation, as the reference does.

## Item-by-item audit against the official repos

Checked RLinf's RoboCasa365 + GR00T path against `robocasa/wrappers/gym_wrapper.py`
(`PandaOmronKeyConverter`), `robocasa/utils/lerobot_utils.py`, and Isaac-GR00T's
`data_config.py`. Aligned items are verified, not assumed.

| item | official | RLinf (after fixes) | verdict |
|---|---|---|---|
| state key names / hdf5 mapping | `body.*`/`hand.*` -> `state.*` | same five keys | aligned |
| state concat order | eef_pos, eef_rot, gripper, base_pos, base_rot | same | **fixed** |
| state rotations | quaternion -> `rotation_6d` | same | **fixed** |
| flat action layout | eef_pos 0:3, eef_rot 3:6, gripper 6, base 7:10, torso 10, mode 11 | same | aligned |
| **gripper / base-mode encoding** | `-1.0 if x < 0.5 else 1.0`, for every policy | was applied **only on the OpenPI branch** | **fixed — this was the bug** |
| action normalization | `binary` for gripper_close and control_mode | was `min_max` | **fixed** |
| executed action steps | `--n_action_steps 16` | was 5 | **fixed** |
| camera identity mapping | `zip(mapped_names, camera_names)`, by name | main/wrist/extra keys resolve to the same three | aligned |
| camera concat order | agentview_left, agentview_right, eye_in_hand | was eye_in_hand first | **fixed** |
| camera resolution | 256x256 | 256x256 | aligned |
| image orientation | `np.copy(img[::-1, :, :])` | `flip_images_vertical`, default True | aligned |
| prompt length | n/a | `padding_value` 570 truncated to 826-token prompt | **fixed** |

### The bug that most plausibly explains the zeros

`rlinf/envs/action_utils.py::prepare_actions_for_robocasa` applied

```python
actions_env[..., 6]  = np.where(actions_env[..., 6]  < 0.5, -1.0, 1.0)   # gripper
actions_env[..., 11] = np.where(actions_env[..., 11] < 0.5, -1.0, 1.0)   # base mode
```

**inside `if SupportedModel(model_type) == SupportedModel.OPENPI:`**. A GR00T policy fell
through to a plain slice, so those two channels reached robosuite as `{0, 1}` where it
expects `{-1, +1}`: the gripper could not open and the mobile base stayed in the wrong
control mode for the whole episode. RoboCasa's own wrapper thresholds both channels at 0.5
for *any* policy -- it is a property of the environment's action schema, not of the model --
so the rule now applies on every path.

## Reference established on this box (2026-08-20)

Per the user's instruction, stopped patching our own eval interface and ran the
**robocasa-benchmark/Isaac-GR00T fork's own evaluation** instead, cloned to
`outputs/reference/Isaac-GR00T` (inside the gitignored `outputs/`, so the workspace repo
does not track it).

```
EP 1 success: False; Cumulative success rate: 0.0
EP 2 success: True;  Cumulative success rate: 0.5
```

Same NVMe checkpoint, same assets, same robosuite, same venv, `--data_config panda_omron
--embodiment_tag new_embodiment --n_action_steps 16 --split target`.

**This rules out the checkpoint, the assets, robosuite, and the machine.** The defect is
entirely in RLinf's glue layer, and there is now a working reference on this box to diff
against — the user's original 0.9 log came from an environment that has since been deleted.

### What the fork's config told us, and what it did not

Its `PandaOmronDataConfig` differs from the NVIDIA `SinglePandaGripperDataConfig` we had
ported in exactly one field:

```python
language_keys = ["annotation.human.task_description"]          # fork
language_keys = ["annotation.human.action.task_description"]   # NVIDIA config we copied
```

Everything else — camera order, the five state keys and their order, action keys, the
min_max/binary normalization modes, the two `rotation_6d` targets — was already identical.
Fixed the language key. **On its own it did not restore success** (still 0/16), so at least
one more difference exists outside the data config.

### Next: differential test, not more field-by-field guessing

Four rounds of "find a plausible discrepancy, fix it, still zero" is enough evidence that
the remaining fault is somewhere I am not thinking to look. With a working reference on the
same box, the efficient move is to dump, for the same observation, what each path actually
feeds the model — key set, state vector, image tensor stats, language string — and what
action comes back. The first tensor that differs is the bug.

## Localised: the defect is RLinf's policy path, not the env (2026-08-20)

Three-way comparison, same checkpoint, same box, `TurnOnMicrowave`:

| policy | env | success_once |
|---|---|---|
| fork `Gr00tPolicy` | fork `RoboCasaGymEnv` | 0.50 (1/2) |
| **fork `Gr00tPolicy`** | **RLinf `Robocasa365Env`** | **0.75 (3/4)** |
| RLinf `predict_action_batch` | RLinf `Robocasa365Env` | 0.00 (0/16) |

**RLinf's environment is fine**, and so is our `convert_robocasa_obs_to_gr00t_format` — the
middle row feeds the fork's policy through *our* converter and still succeeds. The defect is
entirely inside RLinf's policy path: `predict_action_batch` / `get_rl_action` / the RL action
head, everything downstream of `apply_transforms`.

Mechanism, measured earlier: RLinf's four denoising steps move `x` by only 9.5% of its own
magnitude (displacement 0.089 against initial noise 0.937), so the emitted action is ~99%
initial noise — magnitudes of ±1-2.6 where the fork produces 0.01-0.2 deltas.

### Hypotheses tested and exonerated along the way

`language_keys` (fixed anyway), fixed-length eagle padding (816 vs 1024: no effect), the
bf16 round-trip of `states` (0.1298 vs 0.1304 post-transform), action-head weight loading
(314/314 tensors present and matching), the transform pipeline (line-for-line identical to
the fork), the Euler integration (`x0_pred*w0 + x1_pred*w1` verified equal to `x + v·dt`),
the decode/slice order (`CategorySpecificMLP` is strictly positionwise), the flat action
layout, and `prepare_actions` being called on the eval path.

### A previous conclusion is now withdrawn

The 10-step MOPD run reported `velocity_mse` falling 0.0030 -> 0.00086 with cosine rising to
0.89, described at the time as "the distillation is working". **That is not supportable.**
Training ran through the same broken policy path, so student and teacher were both evaluated
on a corrupted velocity field; the loss decreasing only shows two bad outputs converging.
Those numbers should not be reused.

### Plan

Keep the env and the observation converter. Replace the policy side with the fork's
implementation. Note this does **not** fix training on its own: MOPD's `nft_forward` and the
RL action head are the same code family, so the teacher diagnostic must be re-run and pass
before any distillation result is trusted.

## RESOLVED: the pipeline works (2026-08-22, node action00)

First end-to-end evaluation through RLinf's own runner after all four fixes, on a node with
eight free GPUs:

```
atomic_seen teacher, TurnOnMicrowave + CloseToasterOvenDoor, 16 envs, 464-step cap
success_once   = 0.8125   (13/16)
success_at_end = 0.6875
```

The reference implementation scored 0.5-0.75 on the same checkpoint, so this **meets and
exceeds the acceptance bar** — it is not merely non-zero. The teacher is competent on its
own tasks, which is what the whole diagnostic was for.

Environment checks printed at setup, all correct:

```
gr00t resolves to .../gr00t_fork/gr00t/__init__.py    (the benchmark fork, not NVIDIA's)
panda_omron present: True
new_embodiment id: 31                                 (RLinf's own table says 10)
assets: 23G
```

### What was actually wrong, in one place

Four defects, each found by comparing tensors against a reference that was known to
succeed, never by reasoning about the code:

1. **gr00t package mismatch** — the venv had NVIDIA's n1.5-release; these checkpoints were
   trained against `robocasa-benchmark/Isaac-GR00T`, which differs in the eagle processor
   and the state/action transforms.
2. **Prompt truncation** — `rl_head_config.padding_value` 570 against a real 826 tokens,
   silently cutting 232 of 768 image placeholder tokens.
3. **RoPE precision** — `from_pretrained` keeps `inv_freq` in fp32 on purpose; a follow-up
   blanket `model.to(bf16)` quantised those 64 multi-scale constants, corrupting positional
   encoding and leaving `vl_embs` up to 1.5 off per element. Snapshot before the cast and
   restore after; casting back does not recover the precision.
4. **`embodiment_id` 10 vs 31** — the root cause. RLinf patches gr00t's tag table with its
   own (N1.6/N1.7 convention). That id selects which per-embodiment projector weights the
   action head uses, so the model ran on another robot's projectors: index valid, all 314
   weight tensors loaded and matching, every other input tensor bit-identical, no error.

The RL action head itself was never at fault — its velocity matches the reference bit for
bit, so the SDE transitions and transition log-probs that M3/M4 need are intact.

### Next

1. Student (undistilled) baseline through the same harness — needed before any distillation
   number means anything.
2. MOPD training **from scratch**. The earlier 10-step run is withdrawn: it executed with
   defect 4 active, so its falling `velocity_mse` only showed two corrupted outputs
   converging.

## Baseline: all four released checkpoints (2026-08-23, official NVIDIA gr00t package)

Every run `status=OK`, no errors — which also confirms the pipeline works **without** the
benchmark fork (the loader falls back to our local data config when `panda_omron` is
absent). The fork is now optional.

Each teacher on its own suite's two pilot tasks; the student on each suite separately, with
identical settings, so the comparison is apples to apples. 16 envs each, `success_once`:

| suite (2 pilot tasks) | teacher | student | gap |
|---|---|---|---|
| `atomic_seen` (TurnOnMicrowave, CloseToasterOvenDoor) | 0.8125 | **0.75** | +0.0625 |
| `composite_seen` (ScrubCuttingBoard, RinseSinkBasin) | 0.75 | **0.0625** | +0.6875 |
| `composite_unseen` (CuttingToolSelection, CategorizeCondiments) | 0.4375 | **0.0** | +0.4375 |

### These are pilot numbers, not a result

**No conclusion about teacher-vs-student headroom is established by the table above.** Two
tasks per suite, 16 episodes per cell, one seed — and this benchmark's own run-to-run spread
was later measured at three episodes in sixteen on a *fixed* configuration. Gaps of one or
two episodes are unreadable at this size; only the composite gaps (7 and 11 episodes) are
even suggestive, and "suggestive on two tasks" is not a finding.

Settling whether each routed teacher actually beats the student on its assigned tasks needs a
full-scale evaluation over the real task sets with multiple seeds. That has not been run. The
reading below is what the pilot *hints at*, recorded so the design question is not forgotten
— not something to cite.

### What the pilot hints at

**On the atomic pair the teacher offers essentially no headroom**: 0.8125 vs 0.75 is one
episode out of sixteen, well inside the binomial standard error of ~0.11. The student
already does these tasks. Routing to the atomic teacher will contribute almost no learning
signal.

The distillation payoff lives entirely in the two composite suites, where the gaps are
0.75 vs 0.06 and 0.44 vs 0.0.

Consequence for reading any MOPD result on this task set: an improvement will come from the
composite routes, and must **not** be presented as evidence that routing per se works —
with one route contributing nothing, this configuration cannot separate "routing helps"
from "the composite teachers are simply much better".

On the original plan's **gate #2** ("each routed teacher is stronger than the student on its
assigned tasks") the pilot is consistent with the composite suites passing and `atomic_seen`
showing no measurable effect — but at this sample size it **cannot decide the gate either
way**. Treat it as a reason to run the full evaluation, not as its outcome.

### Caveats

Two tasks per suite, n=16, one seed. The atomic pair was chosen for the shortest horizons,
which may have selected precisely the tasks the student already handles; other atomic tasks
could show a real gap. Widening the atomic task set is the cheapest way to find out.

## The delegation is redundant — RLinf's own inference path works (2026-08-23)

Every success number before this point was produced through `predict_action_reference`, the
port of the fork's `Gr00tPolicy.get_action`. MOPD *training* does not use that code, so those
numbers certified nothing about the training path. The A/B had been written but never run.

Run with `MOPD_DISABLE_REF_INFER=1`, same checkpoint, tasks, horizon and 16 envs — the only
variable is the inference path. The switch printed its choice into the log, so this is not
an assumption:

```
(RolloutGroup(rank=2)) [MOPD] inference path = RLinf _get_rl_action (mode=eval, MOPD_DISABLE_REF_INFER=1)
```

| path | success_once | success_at_end |
|---|---|---|
| reference delegation | 0.8125 | 0.75 |
| **RLinf `_get_rl_action`** | **0.875** | **0.8125** |

One episode apart — indistinguishable at n=16, and certainly no worse.

### Why the delegation looked necessary, and why it is not

The two paths are algebraically the same. Expanding `sample_mean_var_val` in eval mode with
`a = t_input`, `d = delta`:

```
x0_pred = x - v·a                    x1_pred = x + v·(1-a)
x_mean  = x0_pred·(1-a-d) + x1_pred·(a+d)
        = x + v·[-a(1-a-d) + (1-a)(a+d)]
        = x + v·d
```

which is exactly the reference loop's `actions = actions + dt * pred_velocity`. In eval mode
`denoise_inds = [-1]*num_steps`, so the stochastic branch never fires and `x_t_std = 0`.

The delegation's docstring claim that the RL path "moves x by only ~9% of its magnitude" was
measured while **defect 1.2/5 (fixed-length eagle padding) was still active on the RLinf path
only** — the reference path never pads. That is the whole of the 0.75-vs-0.00 gap that
motivated the delegation. With padding skipped for RoboCasa365, the gap is gone.

**Consequence:** the delegation can be deleted, and training and evaluation return to a
single code path — which is what makes an evaluation number mean something for training.

### Settled by tensor comparison, not by more episodes

The six-cell A/B left one cell looking suspicious — the student on the atomic pair scored
12/16 through the delegation and 7/16 through RLinf's path:

| cell | delegation | RLinf path | episodes apart |
|---|---|---|---|
| teacher atomic | 0.8125 | 0.875 | +1 |
| teacher composite_seen | 0.75 | 0.625 | −2 |
| teacher composite_unseen | 0.4375 | 0.375 | −1 |
| **student atomic** | **0.75** | **0.4375** | **−5** |
| student composite_seen | 0.0625 | 0.0625 | 0 |
| student composite_unseen | 0.0 | 0.0 | 0 |

Pooled, 45/96 against 38/96 — about one standard error, not significant. But the student-atomic
cell alone is ~1.9 SE, and separating a real difference from noise there would take hundreds
of episodes. `toolkits/mopd/probe_path_equiv.py` settles it in one forward pass instead: same
observations, same seed (both paths draw their initial `x_t` with an identically shaped
`torch.randn` right after the backbone, so seeding immediately before each call makes the
draw identical).

| comparison | relative rms difference | max abs |
|---|---|---|
| the two **code paths**, same seed and observation | **0.038** | 0.057 |
| the **same path**, one different noise seed | **0.424** | 0.672 |

**The two paths differ eleven times less than two legitimate rollouts of one path.** The 3.8%
is the expected bf16-vs-autocast discrepancy — RLinf casts inputs to bf16 by hand where the
reference wrapped the forward in `torch.autocast`. The paths are the same function to
numerical precision, so the student-atomic gap is the initial noise draw, nothing else.

Corroborating: the atomic teacher under one fixed configuration has now been measured five
times at 0.75, 0.8125, 0.875, 0.875 and 0.9375. **A 16-env number moves by three episodes on
its own**, which is most of the gap that looked alarming above.

### Final state, verified end to end

With the delegation deleted **and** the fork removed from the venv, the atomic teacher scores
`success_once = 0.9375`, `success_at_end = 0.875` on NVIDIA's official package — the highest
of the five runs, and comfortably above the reference band of 0.5–0.75.

### The delegation is gone

`predict_action_reference`, `get_action_reference` and the `MOPD_DISABLE_REF_INFER` switch
were deleted (−6.2 KB). Evaluation re-run afterwards on the single remaining path:
`success_once = 0.875`, `success_at_end = 0.8125` — identical to the A/B run. Training and
evaluation now share one code path, which is what makes an evaluation number say anything
about training.

Note: the one-off probes in `outputs/mopd/` that called those methods
(`probe_prep.py`, `probe_refloop.py`, `diff_prep_final.py`, `probe_path_equiv.py`) only run
against the commit before that deletion. They are kept as a record, not as working tools.

### Still unvalidated: the stochastic rollout branch

MOPD rollout does not run the eval branch. With `noise_method: flow_sde` and
`joint_logprob: False`, `get_rl_action(mode="train")` adds SDE noise on one random denoise
step. Measured against the eval action on the same observation:

```
draw 0: rel 0.893    draw 1: rel 0.517    draw 2: rel 0.591
```

That is a large perturbation, and `noise_level` is **0.5**, inherited from the PPO/GRPO
defaults in `examples/embodiment/config/model/gr00t.yaml` where diverse rollouts are the
point. For Method-1 velocity distillation the rollout only decides which states the student
visits, and states drawn 50-90% off the policy's own trajectory may be a poor place to match
a teacher's velocity field. **This value should be revisited before MOPD training** — it was
never chosen for distillation.

## Gate #2, measured per task on the 3-task pilot (2026-08-25)

`success_once`, n=48, horizon 464 (atomic) / 1216 (composite), same settings for every row.
Each teacher on **its own** task — the routed pairing, which a multi-task aggregate cannot
show, because a suite specialist scores poorly on the other suites' tasks by construction.

| task | suite | student | routed teacher | gap |
|---|---|---|---|---|
| TurnOnMicrowave | atomic_seen | 0.4375 | **0.8542** | +20/48 |
| ScrubCuttingBoard | composite_seen | 0.0833 | **0.7500** | +32/48 |
| CuttingToolSelection | composite_unseen | 0.0000 | **0.5208** | +25/48 |

**Gate #2 passes decisively on all three routes.** Every routed teacher is far stronger than
the student on the task it supervises, so distillation toward it has real headroom.

This also settles a question the earlier 6-task pilot could not: those numbers came from
different tasks under a different horizon convention and were never a valid substitute.

### Which makes the first M1 run's collapse a defect, not a bad premise

A 15-step M1 run over these same three tasks degraded the student monotonically:

```
train rollout success_once:  0.25 0.33 0.17 0.25 0.17 0.17 0.083 ... 0.042 0.0 0.021
eval success_at_end:         0.19 -> 0.02
velocity_mse:                0.055 -> 0.033      (falling)
velocity_cosine:             0.974 -> 0.985      (rising)
grad_norm:                   0.54 -> 0.17        (stable, no explosion)
```

It starts at the student's baseline and is destroyed by training, while the loss it is
optimising improves. With the teachers verified stronger on every route, "the teachers are
not worth imitating" is ruled out. Under investigation.

Note `velocity_cosine` starts at **0.974** — student and teachers are already 97% aligned,
both descending from the same pretrained checkpoint. A supervision signal that weak destroying
the policy is itself the anomaly worth explaining.

## Why M1 collapsed, and what the supervision actually contains (2026-08-26)

### The learning-rate sweep

| lr | train-rollout success_once | velocity_mse |
|---|---|---|
| **2e-5** | 0.25 → **0.0** (collapse) | 0.055 → 0.033 (**−40%**) |
| **2e-6** | 0.25–0.375 (stable) | 0.063 → 0.056 (−11%) |
| **2e-7** | 0.17–0.42 (stable) | 0.056 → 0.058 (flat) |

The collapse is a learning-rate effect. But the sweep exposes the real problem: **the rate
that moves the loss destroys the policy, and the rate that spares the policy barely learns.**
At 2e-5, cutting velocity MSE by 40% drove success to zero — reducing this objective was
*anti-correlated* with task success.

### What the objective is actually asking for

`toolkits/mopd/probe_supervision.py`, student vs the atomic teacher on real
TurnOnMicrowave observations, same initial flow noise:

| quantity | value |
|---|---|
| rms(student action) | 0.172 |
| rms(teacher action) | 0.250 |
| **rms(student − teacher)** | **0.223** |
| **same model, different noise draw** | **0.179** |

**The gap between the two policies' actions is the same size as one policy's own variation
across noise draws.** In action space — the space that decides success — the target is
barely outside the model's own output noise.

### The disagreement is concentrated at the late denoise steps

| denoise step | velocity mse | cosine |
|---|---|---|
| 0 | 0.031 | 0.985 |
| 1 | 0.050 | 0.976 |
| 2 | 0.112 | 0.958 |
| **3** | **0.172** | **0.934** |

Step 3 carries **5.5x** the disagreement of step 0, and `p(tau)` is uniform — so **three
quarters of the supervision budget is spent where the two models already agree**, and the one
informative step gets a quarter. Most of the gradient is fitting noise, which is why a 40%
loss reduction bought nothing and cost the policy.

**This is independent support for M3.** Its analytic KL weight, computed earlier from the
flow-SDE coefficients, is `w(tau) = [0.125, 0.211, 0.633, 1.898]` — rising with the denoise
step, 15x from first to last. M3's principled weighting happens to match where the models
actually disagree, which uniform M1 weighting ignores.

### Consequences for the next run

- lr **2e-6** — the highest rate measured stable.
- Far more gradient steps. One update per rollout at ~3.5 min/rollout is the binding cost;
  `update_epoch > 1` reuses the cached teacher velocities, so extra passes cost a student
  forward/backward only. The targets are fixed, so repeating the regression is sound.
- Per-suite evaluation metrics, added to the RoboCasa365 env, because the aggregate is
  dominated by whichever suite the policy is already good at: the student's baselines are
  0.4375 / 0.0833 / 0.0000 on the three pilot tasks, so a gain on a weak suite cannot be
  seen in a single averaged curve.

## M4 on GR00T: first complete run, and a null result (2026-08-28)

M4 had never executed end to end anywhere. The mechanism study's five ManiSkill attempts left
12-line logs stopping at "Connected to Ray cluster", plus one hang and one OOM; a wandb run
`maniskill_m4_openpi_pi05` exists but no local artefact carries a training metric. Five
defects had to be fixed before it would run on GR00T — see
[M4_ENABLEMENT.md](M4_ENABLEMENT.md).

### The run

60 steps, the same three tasks / 24 envs / 48 eval envs / horizon 1216 as the M1 baseline, so
the two are directly comparable. lr 2e-6 and `update_epoch: 1` (the OPD loss is a plain
REINFORCE, `-logprobs * reward`, with no clipping, so reusing a batch is uncorrected
off-policy). `status=OK`, zero errors.

| | result |
|---|---|
| eval `success_once`, 12 validations | 0.271 0.313 0.250 0.333 0.313 0.250 0.313 0.292 0.292 0.313 0.354 0.250 |
| undistilled student baseline | **0.3125** |
| `opd_reward` over 60 steps | −0.033 to −0.053, **no trend** |

Per suite, first to last validation:

| suite | first | last | routed teacher |
|---|---|---|---|
| atomic_seen | 0.750 | 0.750 | 0.854 |
| composite_seen | 0.0625 | 0.000 | 0.750 |
| composite_unseen | 0.000 | 0.000 | 0.521 |

`composite_unseen` was 0.0 in 11 of 12 validations while its teacher scores 0.52.

**Flat, like M1** — but failing differently. M1's loss at least moved under a high learning
rate (it was fitting noise). M4's objective did not move at all in 60 steps.

### Withdrawn diagnostic (sampled-row logging bug)

The analysis below is retained as provenance, not as a valid result. The logger mixed
scored sampled rows with unscored zero-filled rollout rows. It was fixed in `069978cc`, but
the historical run was not repeated with the corrected logger.

`opd_reward = l_T - l_S` broken down by suite (new per-suite logging in
`_log_opd_reward_per_suite`):

| suite | student → teacher success gap | OPD reward mean | sd |
|---|---|---|---|
| atomic_seen | 0.44 → 0.85 (**+0.42**) | −0.54 | 1.77 |
| composite_seen | 0.08 → 0.75 (**+0.67**) | −0.51 | 1.79 |
| composite_unseen | 0.00 → 0.52 (**+0.52**) | −0.49 | 1.72 |

~~**The reward does not track policy quality.** The suite where the student cannot do the task
at all and the teacher succeeds half the time gets the same signal as the suite the student
already half-solves. The three means differ by less than their own step-to-step scatter.

**The signal-to-noise ratio is 0.3.** The standard deviation is 3.4x the mean, and REINFORCE's
gradient is `-logprob * reward`, so the update direction is dominated by noise. `opd_reward`
not moving over 60 steps is not slow learning; the quantity does not carry the information.~~

### Former interpretation (not supported by the withdrawn diagnostic)

Three independent measurements now say the two policies are nearly indistinguishable at the
scale both objectives operate on:

- velocity-field cosine starts at **0.974**;
- in action space they differ by 0.223 rms while the same model under a different noise draw
  differs by **0.179**;
- the per-transition log-prob difference is **uncorrelated with the success gap**.

Both M1 and M4 are single-step objectives. The teacher's advantage appears to live in
long-horizon behaviour, not in any one transition or velocity.

**Not concluded:** whether multi-teacher routing or the mixed-suite batch dilutes the signal.
A single-suite comparison is the next experiment.
