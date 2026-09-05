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

"""Does the new flow_ode branch actually give a deterministic ODE rollout?

Three claims to check, on the real model and real observations:
  1. mode="train" + flow_ode + joint_logprob=True == mode="eval" under the same seed.
     (If flow_ode is right, the train path is the same Euler step the eval path takes.)
  2. Its TRANSITION log-probs are all zero -- get_logprob_norm masks sigma == 0. Entry 0
     is not a transition: under joint_logprob=True both model families prepend
     log N(x_T; 0, I) for the initial noise draw, which is nonzero by construction.
  3. flow_sde still behaves as before: stochastic, and NOT equal to eval.
Plus the guard: flow_ode + joint_logprob=False must raise, not silently misbehave.
"""

import copy
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

REPO = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = Path(
    os.environ.get("ROBOCASA365_CHECKPOINT_ROOT") or os.environ["OPD_MODEL_ROOT"]
)
CKPT = CHECKPOINT_ROOT / "target_posttraining/atomic_seen/checkpoint-60000"
SEED, NENV = 1234, 2

base = OmegaConf.load(f"{REPO}/examples/embodiment/config/env/robocasa365.yaml")
run = OmegaConf.load(f"{REPO}/evaluations/robocasa365/robocasa365_mopd_gr00t_eval.yaml")
whole = OmegaConf.merge(
    OmegaConf.create({"runner": run.runner}),
    OmegaConf.create({"env": {"eval": OmegaConf.merge(base, run.env.eval)}}),
)
OmegaConf.resolve(whole)
cfg = whole.env.eval
cfg.total_num_envs = cfg.num_envs = NENV
cfg.group_size = 1
cfg.task_filter = ["TurnOnMicrowave"]

from rlinf.envs.robocasa365.robocasa365_env import Robocasa365Env  # noqa: E402

env = Robocasa365Env(
    cfg, num_envs=NENV, seed_offset=0, total_num_processes=1, worker_info=None
)
obs, _ = env.reset()

mcfg = OmegaConf.merge(
    OmegaConf.load(f"{REPO}/examples/embodiment/config/model/gr00t.yaml"),
    run.rollout.model,
)
OmegaConf.resolve(mcfg)
mcfg.model_path = str(CKPT)

from rlinf.models import get_model  # noqa: E402


def build(noise_method, joint_logprob):
    c = copy.deepcopy(mcfg)
    with open_dict(c):
        c.rl_head_config.noise_method = noise_method
        c.rl_head_config.joint_logprob = joint_logprob
    m = get_model(c).to("cuda:0")
    m.eval()
    return m


def act(m, mode):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    with torch.no_grad():
        a, res = m.predict_action_batch(copy.deepcopy(obs), mode=mode)
    return a.float().numpy(), res


def rel(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(a**2)), 1e-12))


print("\n########## flow_ode + joint_logprob=True ##########", flush=True)
m = build("flow_ode", True)
a_eval, _ = act(m, "eval")
a_train, res = act(m, "train")
print(f"max |a_train - a_eval| : {np.abs(a_train - a_eval).max():.3e}")
print(f"rel rms diff           : {rel(a_eval, a_train):.3e}")
print(
    f"CLAIM 1 deterministic  : {'PASS' if np.abs(a_train - a_eval).max() == 0 else 'FAIL'}"
)

di = res["forward_inputs"]["denoise_inds"]
lp = res["prev_logprobs"].float().cpu().numpy()
print(f"denoise_inds[0]        : {di[0].tolist()}  (expect arange)")
print(
    f"log_prob entries       : {lp.shape[1]}  (expect num_steps + 1 under joint_logprob)"
)
print(
    f"max |entry 0| (x_T)    : {np.abs(lp[:, 0]).max():.3e}  (nonzero by construction)"
)
print(f"max |entries 1..n|     : {np.abs(lp[:, 1:]).max():.3e}")
print(f"CLAIM 2 zero transitions: {'PASS' if np.abs(lp[:, 1:]).max() == 0 else 'FAIL'}")

# second draw, different seed: a deterministic rollout must still land in the same place
torch.manual_seed(SEED + 7)
torch.cuda.manual_seed_all(SEED + 7)
with torch.no_grad():
    a_train2 = (
        m.predict_action_batch(copy.deepcopy(obs), mode="train")[0].float().numpy()
    )
print(
    f"rel diff across seeds  : {rel(a_train, a_train2):.3e}  "
    f"(nonzero is expected: the INITIAL x_T noise still differs)"
)
del m
torch.cuda.empty_cache()

print("\n########## flow_sde + joint_logprob=False (regression) ##########", flush=True)
m = build("flow_sde", False)
a_eval_s, _ = act(m, "eval")
a_train_s, res_s = act(m, "train")
print(
    f"rel |a_train - a_eval| : {rel(a_eval_s, a_train_s):.4f}  (expect clearly nonzero)"
)
print(
    f"denoise_inds[0]        : {res_s['forward_inputs']['denoise_inds'][0].tolist()}"
    f"  (expect one repeated index)"
)
print(
    f"CLAIM 3 still stochastic: {'PASS' if rel(a_eval_s, a_train_s) > 1e-3 else 'FAIL'}"
)
del m
torch.cuda.empty_cache()

print("\n########## flow_ode + joint_logprob=False must raise ##########", flush=True)
m = build("flow_ode", False)
try:
    act(m, "train")
    print("GUARD: FAIL -- no exception")
except ValueError as exc:
    print(f"GUARD: PASS -- {exc}")

print("\nFLOW_ODE_PROBE_DONE", flush=True)
