# RoboTwin evaluation protocol audit — 2026-11-01

Current training remains unfiltered / seen instructions. Existing calibration and matching pilot evaluation use the same setting. These are not direct replications of published PACE success rates.

## Source evidence

- Pinned RoboTwin `dca9ec682688821c42944eacd7f5dd9bfb15396f`, `script/eval_policy.py`: line200 enables expert_check;224–225 initialize and call play_once;246 accepts only plan_success AND check_success;256 resets the accepted seed before policy evaluation.
- [PACE evaluate.py at fd2e624](https://github.com/niejunnan25/PACE/blob/fd2e624cc19325cc9b20a620c82d51c0d8e25cc1/pace/scripts/evaluate.py): line31 selects unseen language;175–207 perform screening;218 resets before policy. This evaluation source revision is distinct from the recorded teacher training/config source.
- Bundled RLinf train/eval seed lists cover all4tasks, but lack embodiment/config/revision provenance. They may supply candidates, not certified compatible pools.

## Matched comparisons

1. Keep the ongoing unfiltered training and its matching base/student evaluation.
2. Add a separate small faithful-protocol evaluation: locally validate candidate seeds under the pinned Aloha-clean setup using the scripted expert, store accepted seed + episode_info + fixed unseen instruction + config/source hashes, discard demonstration trajectories.
3. Evaluate learned teacher, base, and trained student from the exact same reset and prompt. Policy failures stay in the denominator; do not screen by learned-policy success.
4. Report counts, language split, actual seeds, preparation cost and protocol explicitly. Five episodes/task is a pilot, not a formal benchmark estimate.

This protocol difference may affect scores, but the current5episode teacher samples are too small to attribute the published-score gap to it. No changes to the active training are implied.
