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

"""Is the velocity-MSE objective a useful signal for this student/teacher pair?

The LR sweep left a contradiction: at 2e-5 velocity_mse falls 40% and the policy dies;
at 2e-6 the policy survives and the loss barely moves. Before choosing new
hyperparameters, measure what the objective actually asks the student to change.

Three numbers, on real observations:
  1. how far apart the student's and teacher's EMITTED ACTIONS are -- the thing that
     decides task success;
  2. how far apart their VELOCITY FIELDS are at the student's own flow states -- the
     thing the loss minimises;
  3. how much of (1) a perfect fix of (2) would actually buy.

If the action gap is small, there is nothing to learn and the loss is chasing noise.
If it is large, the objective has real headroom and the problem is optimisation.
"""

import copy
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = Path(
    os.environ.get("ROBOCASA365_CHECKPOINT_ROOT") or os.environ["OPD_MODEL_ROOT"]
)
STUDENT = CHECKPOINT_ROOT / "pretraining/checkpoint-80000"
TEACHER = CHECKPOINT_ROOT / "target_posttraining/atomic_seen/checkpoint-60000"
NENV, SEED = 8, 1234

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
print(f"env ready: {NENV} envs on TurnOnMicrowave", flush=True)

mcfg = OmegaConf.merge(
    OmegaConf.load(f"{REPO}/examples/embodiment/config/model/gr00t.yaml"),
    run.rollout.model,
)
OmegaConf.resolve(mcfg)

from rlinf.models import get_model  # noqa: E402
from rlinf.models.embodiment.base_policy import ForwardType  # noqa: E402


def load(path):
    c = copy.deepcopy(mcfg)
    c.model_path = path
    m = get_model(c).to("cuda:0")
    m.eval()
    return m


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


student, teacher = load(str(STUDENT)), load(str(TEACHER))


# --- 1. emitted actions, same seed so the initial flow noise matches ---
def act(m):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    with torch.no_grad():
        return (
            m.predict_action_batch(copy.deepcopy(obs), mode="eval")[0].float().numpy()
        )


a_s, a_t = act(student), act(teacher)
print("\n=== 1. emitted action gap (what decides success) ===", flush=True)
print(f"rms(student)          {rms(a_s):.4f}")
print(f"rms(teacher)          {rms(a_t):.4f}")
print(f"rms(student-teacher)  {rms(a_s - a_t):.4f}")
print(f"relative gap          {rms(a_s - a_t) / max(rms(a_t), 1e-9):.1%}")

# scale reference: two rollouts of the SAME model, different flow noise
torch.manual_seed(SEED + 1)
torch.cuda.manual_seed_all(SEED + 1)
with torch.no_grad():
    a_s2 = (
        student.predict_action_batch(copy.deepcopy(obs), mode="eval")[0].float().numpy()
    )
print(
    f"same model, new noise {rms(a_s - a_s2) / max(rms(a_s), 1e-9):.1%}  <- the noise floor"
)

# --- 2. velocity gap at the student's own flow states ---
print(
    "\n=== 2. velocity gap at the student's flow states (what the loss sees) ===",
    flush=True,
)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
with torch.no_grad():
    _, res = student.predict_action_batch(copy.deepcopy(obs), mode="train")
fi = res["forward_inputs"]
chains = fi["chains"]  # [B, num_steps+1, horizon, dim]
nsteps = student.action_head.num_inference_timesteps
for k in range(nsteps):
    x_t = chains[:, k]
    idx = torch.full((x_t.shape[0],), k, dtype=torch.long)
    vs, vt = [], []
    for m, out in ((student, vs), (teacher, vt)):
        with torch.no_grad():
            o = m(
                forward_type=ForwardType.NFT,
                forward_inputs=fi,
                nft_inputs={"x_t": x_t, "timesteps": idx},
                compute_values=False,
            )
        out.append(o["v_theta"].float().cpu().numpy())
    v_s, v_t = vs[0], vt[0]
    cos = float(
        np.mean(
            np.sum(v_s * v_t, axis=(1, 2))
            / (
                np.linalg.norm(v_s, axis=(1, 2)) * np.linalg.norm(v_t, axis=(1, 2))
                + 1e-9
            )
        )
    )
    print(
        f"  step {k}:  mse {np.mean((v_s - v_t) ** 2):.4f}   cosine {cos:.4f}   "
        f"|v_s| {rms(v_s):.3f}  |v_t| {rms(v_t):.3f}"
    )

print("\nPROBE_DONE", flush=True)
