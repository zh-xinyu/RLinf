# RoboTwin four-task checkpoint preparation

This directory downloads the public specialists and converts them into the pinned RLinf/OpenPI PyTorch implementation. It does not train teachers or run PACE's execution-horizon algorithm.

## Source and interface contract

- Teachers: `niejunnan25/pi05-robotwin`, revision `17ef545270cf26ed81169893b60f845f80ce6557`. Tasks: `adjust_bottle`, `handover_mic`, `place_container_plate`, `stack_bowls_two`. Downloads verify the repository manifest's SHA256 for every file.
- Student: official `gs://openpi-assets/checkpoints/pi05_base`. The saved `gcs_source_manifest.json` fixes every object's generation; replays reuse that manifest. Files verify GCS MD5 or CRC32C for composite objects.
- Converter/model source: RLinf/OpenPI revision `c5dc4b9296a1a4739bf52828f28a579f12dce763`, including its required `transformers_replace` overlay on Transformers 4.53.2.
- Pi0.5, action dimension 32 with 14 physical dimensions, predicted horizon 50, max prompt length 200. Initial policy inference uses 10 Euler steps and executes 25 actions per chunk.
- **`adapt_to_pi=False`**, matching the released PACE task configuration. Joint deltas apply to dimensions 0–5 and 7–12; grippers stay absolute. Retain each specialist's original quantile normalization, supplied to both that teacher and the task-conditioned preprocessing of the shared student.
- Camera inputs: `cam_high`, `cam_left_wrist`, `cam_right_wrist`; task language prompt and 14-dimensional state before model padding. The tokenizer is the checkpoint repository's unmodified PaliGemma tokenizer.

The author-validated official OpenPI revision `2b1f3779ec2e3ddb56fd1425944e5b3d084c28ea` and the pinned RLinf fork have byte-identical `policies/aloha_policy.py`, `models/tokenizer.py`, `models/pi0.py`, and `transforms.py`. RLinf's RoboTwin bridge uses the same joint-delta mask. Task normalization must not be silently replaced with joint or other-teacher statistics.

## Commands

Run from this worktree after activating the isolated runtime. `MODEL_ROOT` below is a writable model storage directory, separate from RoboCasa assets.

```bash
export MODEL_ROOT=/localscratch2/liyifa11/XyFile/opd-robotwin4/models
python toolkits/robotwin_mopd/checkpoints/download.py pace --root "$MODEL_ROOT/pi05_pace_jax"
python toolkits/robotwin_mopd/checkpoints/download.py base --root "$MODEL_ROOT/pi05_base_jax"

# On another machine, pass the recorded manifest to reproduce the same base:
# python .../download.py base --root "$MODEL_ROOT/pi05_base_jax" --base-manifest /path/to/gcs_source_manifest.json

JAX_PLATFORMS=cpu USE_TF=0 python toolkits/robotwin_mopd/checkpoints/convert.py \
  --source "$MODEL_ROOT/pi05_base_jax" --destination "$MODEL_ROOT/pi05_base_torch" \
  --openpi-root "$OPENPI_PATH"

for task in adjust_bottle handover_mic place_container_plate stack_bowls_two; do
  JAX_PLATFORMS=cpu USE_TF=0 python toolkits/robotwin_mopd/checkpoints/convert.py \
    --source "$MODEL_ROOT/pi05_pace_jax/$task" \
    --destination "$MODEL_ROOT/pi05_pace_torch/$task" --openpi-root "$OPENPI_PATH"
done
mkdir -p "$OPENPI_DATA_HOME/big_vision"
cp "$MODEL_ROOT/pi05_pace_jax/tokenizer.model" "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
```

`convert.py` checks all source-to-model keys, corrects upstream asset copying to preserve checkpoint-local statistics, and writes a weight SHA256 plus conversion provenance. It permits only the verified tied PaliGemma embedding/head alias and the unused action expert language head absent from JAX. The latter is initialized by PyTorch and never participates in velocity/action prediction. Converted files contain it; use `safetensors.torch.load_model(..., strict=True)` to restore the shared embedding alias correctly.

Final teacher normalization files live at `$MODEL_ROOT/pi05_pace_torch/<task>/assets/robotwin/norm_stats.json`. Float32 converted files are approximately 14.47 GB each. The five models and source checkpoints require separate disk space.

## Numerical validation performed on 2026-11-01

All five conversions passed the parameter audit. Base and `adjust_bottle` passed strict safetensors loading and GPU inference at three flow timesteps plus a complete 10-step trajectory, with finite outputs.

`verify_inference.py` uses a deterministic synthetic normalized observation, identical tokenizer IDs and identical Gaussian noise in separate JAX/PyTorch processes. It measures arithmetic/model conversion fidelity, **not closed-loop task success or an end-to-end observation adapter test**. `adjust_bottle` float32 results:

| Quantity | Relative RMSE | Maximum absolute error |
| --- | ---: | ---: |
| Velocity at t=1.0 | 0.0838% | 0.00352 |
| Velocity at t=0.5 | 0.0880% | 0.00683 |
| Velocity at t=0.1 | 0.0791% | 0.01111 |
| Final normalized action, 10 Euler steps | 0.1304% | 0.00128 |

Logs, `.npz` outputs and `adjust_equivalence.json` are in the root workspace's `outputs/robotwin4/setup/checkpoints/`. The simulator/teacher calibration remains a separate validation step. Source model terms continue to apply; the tool license does not replace the upstream model terms.
