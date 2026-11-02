# Fixed author-cache evaluation pools

`author_pool.py` creates schema 2 / `author_cache_replay_unseen` pools from the
pinned PACE release. These are **author-accepted cached scenes**, not scenes
re-screened by a local scripted expert. No teacher performance is consulted when
selecting scenes. The existing schema 1 expert-screened protocol remains valid.

Activate the verified RoboTwin4 runtime first. Pool construction is CPU-only and
requires the pinned original reference checkout, all 54 cache compatibility
hashes, an unchanged author language directory, and the verified SAPIEN 3.0.0b1
profile. The reference checkout may retain its asset symlink.

From the isolated worktree, with `PACE_REFERENCE_ROOT` set to that checkout:

```bash
python -m toolkits.robotwin_mopd.author_pool \
  --reference-root "$PACE_REFERENCE_ROOT" \
  --cohort 0 --start 0 --count 20 --purpose development \
  --output /absolute/path/author_dev20.json

python -m toolkits.robotwin_mopd.author_pool \
  --reference-root "$PACE_REFERENCE_ROOT" \
  --cohort 1 --start 50 --count 50 --purpose formal_heldout \
  --output /absolute/path/author_formal50.json
```

These select each task's first 20 scenes from author cohort 0 and last 50 scenes
from author cohort 1. The formal pool is **reserved until parameters are frozen**;
creating it is not authorization to inspect formal policy outcomes. Pools retain
cohort/order/count, cache/revision/source hashes, local runtime/config/horizons,
and fixed unseen prompts. Author language generation uses 100 candidates and a
scene-seeded Python RNG; generation restores the caller's RNG state.

Evaluate a development task through the production PyTorch policy:

```bash
CUDA_VISIBLE_DEVICES=0 python -m toolkits.robotwin_mopd.evaluate \
  --task adjust_bottle --checkpoint teacher --episodes 20 \
  --evaluation-pool /absolute/path/author_dev20.json \
  --output /absolute/path/dev_adjust_teacher.json
```

Before loading model weights, evaluation checks pool content, exact ordered
scene selection, prompt generation, config/horizons, local source files, and the
runtime profile. A fixed reset cannot retry another seed. Existing evaluation
noise remains `torch.manual_seed(actual_scene_seed)` once per episode, followed
by the production policy's normal noise draws; this is not the explicit NumPy
noise protocol used by the simulator diagnostic. Outputs record the distinction.

Absolute installation paths remain provenance only. Runtime validation compares
versions, content hashes and the vendor wheel hash; an identical installation at
another path is accepted. Set `PACE_REFERENCE_ROOT` to the relocated pinned
author checkout. Local source/config changes invalidate the pool. No online expert feasibility revalidation is performed during evaluation.
