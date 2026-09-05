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

"""Are the two inference paths the same function? Compare tensors, not success rates.

The six-cell A/B left one suspicious cell (student on atomic: 12/16 through the delegation
vs 7/16 through RLinf's path). Separating a real difference from rollout noise there would
take hundreds of episodes. Comparing the emitted action for the SAME observation under the
SAME seed settles it in one forward pass.

Both paths draw their initial x_t with an identically-shaped torch.randn immediately after
the backbone forward, so seeding right before each call makes the noise draw identical and
any remaining difference is the code, not the sampling.

Also checks the branch the A/B never touched: get_rl_action(mode="train"), which adds SDE
noise on one random denoise step and is what MOPD rollout actually runs.
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
CKPT = CHECKPOINT_ROOT / "target_posttraining/atomic_seen/checkpoint-60000"
SEED = 1234
NENV = 4

base = OmegaConf.load(f"{REPO}/examples/embodiment/config/env/robocasa365.yaml")
run = OmegaConf.load(f"{REPO}/evaluations/robocasa365/robocasa365_mopd_gr00t_eval.yaml")
whole = OmegaConf.merge(
    OmegaConf.create({"runner": run.runner}),
    OmegaConf.create({"env": {"eval": OmegaConf.merge(base, run.env.eval)}}),
)
OmegaConf.resolve(whole)
cfg = whole.env.eval
cfg.total_num_envs = NENV
cfg.num_envs = NENV
cfg.group_size = 1
cfg.task_filter = ["TurnOnMicrowave", "CloseToasterOvenDoor"]

from rlinf.envs.robocasa365.robocasa365_env import Robocasa365Env  # noqa: E402

env = Robocasa365Env(
    cfg, num_envs=NENV, seed_offset=0, total_num_processes=1, worker_info=None
)
obs, _ = env.reset()
print(f"env ready: {NENV} envs, states {tuple(obs['states'].shape)}", flush=True)

mcfg = OmegaConf.merge(
    OmegaConf.load(f"{REPO}/examples/embodiment/config/model/gr00t.yaml"),
    run.rollout.model,
)
OmegaConf.resolve(mcfg)
mcfg.model_path = str(CKPT)

from rlinf.models import get_model  # noqa: E402

model = get_model(mcfg).to("cuda:0")
model.eval()


def seeded(fn, *a, **kw):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    return fn(*a, **kw)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


# predict_action_batch mutates env_obs["states"] (deliberate bf16 round-trip), so each call
# gets its own copy -- otherwise the second call sees the first call's damage.
with torch.no_grad():
    os.environ["MOPD_DISABLE_REF_INFER"] = "1"
    a_ref = seeded(model.predict_action_reference, copy.deepcopy(obs)).float().numpy()
    a_rl = (
        seeded(model.predict_action_batch, copy.deepcopy(obs), mode="eval")[0]
        .float()
        .numpy()
    )

print("\n=== eval: reference delegation vs RLinf _get_rl_action ===", flush=True)
print(f"shape           {a_ref.shape}", flush=True)
print(f"rms(ref)        {rms(a_ref):.6f}", flush=True)
print(f"rms(rl)         {rms(a_rl):.6f}", flush=True)
d = np.abs(a_ref - a_rl)
print(f"max |diff|      {d.max():.6f}", flush=True)
print(f"mean |diff|     {d.mean():.6f}", flush=True)
print(f"rel rms diff    {rms(a_ref - a_rl) / max(rms(a_ref), 1e-12):.6f}", flush=True)
print(f"ref[0,0,:6]     {np.round(a_ref[0, 0, :6], 5).tolist()}", flush=True)
print(f"rl [0,0,:6]     {np.round(a_rl[0, 0, :6], 5).tolist()}", flush=True)

# Scale reference: how far apart are two *legitimately different* rollouts of the same path?
with torch.no_grad():
    torch.manual_seed(SEED + 1)
    torch.cuda.manual_seed_all(SEED + 1)
    a_ref2 = model.predict_action_reference(copy.deepcopy(obs)).float().numpy()
print("\n--- scale check: same path, different noise draw ---", flush=True)
print(f"max |diff| ref vs ref'   {np.abs(a_ref - a_ref2).max():.6f}", flush=True)
print(
    f"rel rms diff             {rms(a_ref - a_ref2) / max(rms(a_ref), 1e-12):.6f}",
    flush=True,
)

# The branch the six-cell A/B never exercised: MOPD rollout runs mode="train".
print("\n=== train mode (flow_sde, one stochastic step) vs eval ===", flush=True)
with torch.no_grad():
    for k in range(3):
        torch.manual_seed(SEED + 100 + k)
        torch.cuda.manual_seed_all(SEED + 100 + k)
        a_tr = (
            model.predict_action_batch(copy.deepcopy(obs), mode="train")[0]
            .float()
            .numpy()
        )
        print(
            f"  draw {k}: rms={rms(a_tr):.6f}  "
            f"max|a_train - a_eval|={np.abs(a_tr - a_rl).max():.6f}  "
            f"rel={rms(a_tr - a_rl) / max(rms(a_rl), 1e-12):.6f}",
            flush=True,
        )

print("\nPATH_EQUIV_DONE", flush=True)
