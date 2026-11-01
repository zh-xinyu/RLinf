# RoboTwin4 isolated pilot runtime

This installation uses the existing RLinf installer helpers, a separate Python
environment, pinned source revisions and official RoboTwin2 assets. It does not
modify the RoboCasa runtime. The pilot uses RGB observations and MPLib; optional
cuRobo and point-cloud/PyTorch3D operations are not installed.

Run from this worktree with `uv`, Git, NVIDIA drivers and Vulkan ICD available:

```bash
export ROBOTWIN_RUNTIME_ROOT=/localscratch2/liyifa11/XyFile/opd-robotwin4
bash toolkits/robotwin_mopd/setup/provision_runtime.sh
bash toolkits/robotwin_mopd/setup/finalize_runtime.sh
source "$ROBOTWIN_RUNTIME_ROOT/env.sh"
python toolkits/robotwin_mopd/setup/download_assets.py
(cd "$ROBOTWIN_PATH" && python script/update_embodiment_config_path.py)
CUDA_VISIBLE_DEVICES=7 python toolkits/robotwin_mopd/setup/smoke_simulator.py
```

`ASSETS_PATH` points to the RoboTwin repository, which contains `assets/`.
`write_environment.sh` records the current worktree and runtime root explicitly;
rerun it after moving either directory. The default Vulkan ICD path is specific
to this server and can be overridden after sourcing `env.sh`.

Verified on action00: Python 3.11.14, PyTorch 2.6.0+cu124, transformers 4.53.2,
JAX 0.5.3, Orbax 0.11.13, SAPIEN 3.0.1, NumPy 1.26.4. OpenPI's documented
`transformers_replace` overlay is required and applied by the finalizer.
Hydra 1.3.2 preserves the repository's `version_base="1.1"` entrypoints;
PyArrow 19.0.1 supports NumPy 1.26. Ray 2.56.1 matches the existing runtime.

Source revisions:

- RoboTwin `dca9ec682688821c42944eacd7f5dd9bfb15396f`.
- RLinf/OpenPI `c5dc4b9296a1a4739bf52828f28a579f12dce763`.
- `TianxingChen/RoboTwin2.0` assets `3dc3b798668feb99ac61cc9086d84cbcc3d79186`.

The runtime root contains `source_revisions.txt`, `base_freeze.txt` and
`runtime_freeze.txt` for exact installed-package inspection. The pins file covers
known compatibility boundaries; it is not a complete dependency lock.

Validation: actual headless rendering, Aloha `adjust_bottle` reset, three RGB
camera observations, 14-dimensional state and one native `take_action` passed.
This tests simulator integration, not policy success. Upstream reset can retry
unstable seeds; the multitask adapter records the resulting seed. Initial setup
logs are under the main workspace's `outputs/robotwin4/setup/runtime/`.

Some optional TensorFlow imports emit a NumPy C-API warning; OpenPI imports and
the targeted model unit tests still pass. No flash-attn build is installed;
the current OpenPI path uses eager attention.

For full-parameter NFT training with gradient checkpointing, set
`actor.model.openpi.nft_joint_forward: true`. Training then evaluates the given
flow state and time in one prefix/suffix attention pass, preserving gradients to
the visual/language backbone without a training KV cache. Teacher evaluation
retains the cached split forward. This flag defaults to false for existing
recipes. Tiny real Gemma tests compare both velocity and parameter gradients,
and verify a finite optimizer update with checkpointing enabled.
