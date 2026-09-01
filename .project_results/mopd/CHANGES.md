# What was changed to make RoboCasa365 + GR00T N1.5 work in RLinf

Branch `opd-mopd-routed`. Written 2026-08-22, after the teacher diagnostic returned
`success_once = 0.8125` (13/16) against the reference implementation's 0.5–0.75.

**This is Path A**: RLinf's own model class, action head and environment are used
throughout. No `Gr00tPolicy` object from the reference fork is instantiated anywhere in the
production path — the fork enters only as the installed `gr00t` *package* and as the source
of the `panda_omron` data config. See §4 for the one place where that statement needs a
qualifier.

---

## 1. The four defects, and the fix for each

Every one was located by fixing all inputs and comparing one tensor at a time against a
reference run that was known to succeed — never by reasoning about the code.

### 1.1 gr00t package mismatch *(environment, not code — and probably NOT a defect)*

The venv had NVIDIA's `Isaac-GR00T` n1.5-release; the checkpoints were trained against
`robocasa-benchmark/Isaac-GR00T`. Swapping to the fork was the first thing tried, and the
first thing that made the pipeline run.

**This was listed as a defect on weak evidence, and the evidence has since gone the other
way.** The three files that matter (`gr00t_n1.py`, `transforms.py`, `embodiment_tags.py`)
diff to **zero lines**, and both packages map `new_embodiment` to 31. On 2026-08-23 the whole
diagnostic was re-run against **NVIDIA's official package** and all four checkpoints
evaluated successfully (see §6) — so the fork was never required. Defects 1.2–1.4 are the
real ones.

**Current state:** the fork is **deleted** (2026-08-23) — from the venv and from
`outputs/reference/`. NVIDIA's official `n1.5-release`, which `install.sh` installs anyway,
is the only gr00t package. The loader still prefers a `panda_omron` data config if one is
present and otherwise uses our `RoboCasaPandaOmronDataConfig`, verified **16/16 fields
identical** to the fork's while the fork was still available to compare against.
Rationale for keeping the official one: the fork is third-party and could disappear, which
would make the environment unreproducible.

*File:* `rlinf/models/embodiment/gr00t/gr00t_n1d5/__init__.py`

### 1.2 Prompt truncation — `rl_head_config.padding_value`

`predict_action_batch` pads `eagle_input_ids` to a fixed length with `F.pad`, and a negative
pad amount **truncates**. RoboCasa's three cameras tokenize to 826 tokens; the LIBERO-derived
default of 570 silently cut 232 of the 768 image placeholder tokens, and the backbone
discarded that many vision embeddings.

**Initial fix:** `padding_value: 1024` in the config removed truncation. A later direct
comparison showed that fixed padding itself contaminated tokens because this GR00T path did
not consume the attention mask. The durable fix is to skip fixed-length padding for
RoboCasa365 and use the processor's longest-in-batch length. The config value remains for
historical compatibility but is inert on this path.

*Files:* `examples/embodiment/config/robocasa365_mopd_gr00t.yaml`,
`rlinf/models/embodiment/gr00t/gr00t_n1d5/gr00t_action_model.py`

### 1.3 RoPE precision — `inv_freq` quantised to bf16

`from_pretrained` applies `torch_dtype` per module and deliberately keeps the rotary
embedding's inverse frequencies in fp32. A follow-up blanket `model.to(torch_dtype)`
overrode that. Those 64 constants span several orders of magnitude, so bf16's ~3 significant
digits corrupt the positional encoding — progressively worse at later positions. Over an
816-token sequence the vision-language embeddings ended up **1.5 off per element**.

**Fix:** snapshot the buffers *before* the cast and restore them after. Casting bf16 back to
fp32 does **not** recover the precision (0.80584 → 0.8046875 is permanent) — an earlier
attempt that did exactly that changed nothing, which is worth remembering.

*File:* `rlinf/models/embodiment/gr00t/gr00t_n1d5/__init__.py`

### 1.4 `embodiment_id` 10 vs 31 — the root cause

RLinf patches `gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING` with its own table, which
follows the N1.6/N1.7 convention and maps `new_embodiment` to **10**. These checkpoints were
trained with **31**.

That id selects which per-embodiment projector weights the action head uses — `state_encoder`,
`action_encoder` and `action_decoder` are all `CategorySpecificMLP`, indexed by it. So the
model ran on **another robot's projectors**: the index is valid, all 314 weight tensors load
and match, every other input tensor is bit-identical, and nothing raises.

**Fix:** skip that patch for `new_embodiment` / `robocasa_panda_omron` and let the installed
package's own mapping stand.

*File:* `rlinf/models/embodiment/gr00t/gr00t_n1d5/__init__.py`

---

## 2. New glue that had to be written (this combination did not exist)

RLinf ships RoboCasa365 (with OpenPI) and GR00T (with LIBERO / ManiSkill / IsaacLab). Their
intersection did not exist: `install_gr00t_model` rejected `--env robocasa365`, there was no
RoboCasa observation converter, and the loader knew only three embodiment tags.

| what | file |
|---|---|
| RoboCasa observation converter (3 cameras, 16-d state split) and the 12-d action converter | `rlinf/models/embodiment/gr00t/simulation_io.py` |
| `new_embodiment` / `robocasa_panda_omron` branch, using the fork's `panda_omron` config | `rlinf/models/embodiment/gr00t/gr00t_n1d5/__init__.py` |
| `--model gr00t --env robocasa365` install support | `requirements/install.sh` |
| training and evaluation configs | `examples/embodiment/config/robocasa365_mopd_gr00t.yaml`, `evaluations/robocasa365/robocasa365_mopd_gr00t_eval.yaml` |

The data config was originally a hand-written port of NVIDIA's `SinglePandaGripperDataConfig`.
Reconstructing it by inference was the mistake that started this whole investigation. It was
then replaced by the fork's `panda_omron`, and now sits behind a **fallback**: use
`panda_omron` if the installed gr00t exposes it, otherwise our
`RoboCasaPandaOmronDataConfig` — whose nine fields were checked programmatically against
`panda_omron`, field by field, rather than reasoned about. That fallback is what lets the
official package work.

---

## 3. Changes for MOPD itself (routing), unrelated to the bugs above

| what | file |
|---|---|
| `ForwardType.NFT` → velocity at an explicit `(x_t, denoise index)`, by splitting `get_velocity` out of `sample_mean_var_val` so teacher and student are scored by the same code | `gr00t_n1d5/gr00t_action_model.py` |
| per-sample `task_groups` label carried through `EnvOutput`'s fixed observation contract | `rlinf/data/embodied_io_struct.py`, `rlinf/envs/robocasa365/robocasa365_env.py` |
| each task remembers which soup it came from (several soups previously collapsed to a joined label) | `rlinf/envs/robocasa365/utils.py` |
| rollout tags each sample with the teacher its suite routes to | `rlinf/workers/rollout/hf/huggingface_worker.py` |
| velocity distillation generalised beyond OpenPI; teacher forward partitioned by suite so only one teacher sits on the GPU at a time | `rlinf/workers/actor/fsdp_actor_worker.py` |
| gripper / base-mode binarisation applies on every model path, not only OpenPI's | `rlinf/envs/action_utils.py` |
| GR00T usable without a value head; `eval()` returns self | `gr00t_n1d5/gr00t_action_model.py` |

---

## 4. Path A with no qualifier (as of 2026-08-23)

This section used to record that evaluation did **not** run RLinf's own inference path: it
delegated to `predict_action_reference`, a port of the fork's `Gr00tPolicy.get_action`. That
delegation has been removed, along with `get_action_reference` and the
`MOPD_DISABLE_REF_INFER` switch.

It was added while the model still produced noise, and it was pointed at the wrong cause. The
reference path never applies fixed-length eagle padding; RLinf's path did. That was defect
1.2/5 — the whole of the 0.75-vs-0.00 gap the delegation was built to work around. Once
padding was skipped for RoboCasa365, the two paths were equivalent, and the note in the
delegation's docstring that the RL path "moves x by only ~9% of its magnitude" described a
measurement taken while that defect was still live.

Evidence for removing it, in increasing order of strength:

1. **Algebra.** Expanding `sample_mean_var_val` in eval mode gives exactly the reference
   loop's Euler step, `x + v·dt`; in eval `denoise_inds = [-1]*num_steps`, so the stochastic
   branch never fires and `x_t_std = 0`.
2. **Six matched evaluation cells** (three teachers, student on three suites): 45/96 versus
   38/96 successes, about one standard error.
3. **Direct tensor comparison** under a shared seed: the two paths differ by 3.8% relative
   rms while two rollouts of the *same* path differ by 42.4% — an eleven-fold margin. This is
   the decisive one; success rates at n=16 cannot resolve a difference this small.
4. **Re-run after deletion:** `success_once = 0.875`, `success_at_end = 0.8125`, matching the
   pre-deletion A/B exactly.

Training and evaluation are back on one code path, which is the property that lets an
evaluation number certify the training path.

**What this still does not cover:** MOPD rollout runs `mode="train"`, whose SDE noise branch
these evaluations never exercise. See RESULTS.md — `noise_level: 0.5` is inherited from the
PPO/GRPO defaults and needs revisiting before distillation.

---

## 5. What was deliberately **not** changed

The RL action head keeps everything M3/M4 need: `sample_mean_var_val` (SDE transitions),
`get_logprob_norm` (transition log-probs), `denoise_inds`, `joint_logprob` / `noise_method`
config. Its velocity was measured bit-identical to the reference's, so it was never at fault.
An earlier suggestion of mine to delete it in favour of the stock head **would** have cost
M3/M4 support and was dropped.

---

## 6. Status of results

**Verified 2026-08-23, official NVIDIA gr00t package, all four released checkpoints, every
run `status=OK`** (each teacher on its own suite's two pilot tasks; student on each suite
separately with identical settings; 16 envs each; `success_once`):

| suite (2 pilot tasks) | teacher | student | gap |
|---|---|---|---|
| `atomic_seen` | 0.8125 | 0.75 | +0.0625 |
| `composite_seen` | 0.75 | 0.0625 | +0.6875 |
| `composite_unseen` | 0.4375 | 0.0 | +0.4375 |

The atomic teacher's margin is **one episode in sixteen** — inside the binomial standard
error of ~0.11. On that pair the student already succeeds, so routing to the atomic teacher
offers almost nothing to distil, and the pilot's headroom lives entirely in the two composite
suites. Any improvement this configuration produces must not be read as evidence that routing
per se works. Detail and caveats: [RESULTS.md](RESULTS.md).

**Withdrawn:** the earlier 10-step MOPD run (`velocity_mse` 0.0030 → 0.00086). It executed
with defect 1.4 active, so student and teacher both ran on the wrong projectors and the
falling loss meant nothing.

**Done since:** the A/B was run and the delegation removed (§4); the fork was deleted and
the official package is now the only gr00t.

**Not yet done:**
- validating `get_rl_action(mode="train")`, the branch MOPD rollout actually runs, and
  choosing a `noise_level` for distillation rather than inheriting the PPO/GRPO default of 0.5
- MOPD training from scratch
