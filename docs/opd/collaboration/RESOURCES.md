# Experiment resources and transfer

Git and the [fixed profile](../../../OPD_ENVIRONMENT.md) provide the software.
Models and simulator assets are separate inputs.
[checkpoints.json](checkpoints.json) records the SHA256 and size of every model
file loaded by this experiment. These are **input** checkpoints; trained output
checkpoints are not part of the collaborator's required deliverables.

## Three teachers

Official repository: [robocasa/robocasa365_checkpoints](https://huggingface.co/robocasa/robocasa365_checkpoints).
Pinned revision: `c484448aba1a9b60a04c9b0ca117241518ea69f3`.
Download only the three teachers' inference files:

```bash
python -m toolkits.robocasa_mopd.resources download-teachers \
  --destination /absolute/path/teachers
export OPD_MODEL_ROOT=/absolute/path/teachers/gr00t_n1-5/foundation_model_learning
```

The expected layout is
`target_posttraining/{atomic_seen,composite_seen,composite_unseen}/checkpoint-60000`.
Each teacher must retain its own `config.json` and
`experiment_cfg/metadata.json`. Do not substitute the generic NVIDIA GR00T
checkpoint or share normalization metadata across teachers. The manifest was
checked against this HF revision, using LFS SHA256 for weight files.

## C10 input

C10 is `checkpoint-60000` from joint SFT on 10% of demonstrations across the
three task suites. It is separate from the released specialist teachers.
All four experiments start with these weights and a fresh optimizer; they do
not restore the SFT optimizer, scheduler, or RNG state.

```text
checkpoint-60000/
  config.json
  model.safetensors.index.json
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
  experiment_cfg/metadata.json
```

The project owner supplies the download location separately. Google Drive,
rsync/scp, or Hugging Face are all acceptable; verify contents against the
manifest rather than trusting a directory name. Drive is sufficient for a
one-time transfer. Repeated deployments can use a pinned HF revision plus
content verification. For restricted HF collaboration, use an organization
repository accessible to both parties; do not share personal access tokens.
The repository does not upload C10 or store download credentials.

## Bind the shared teacher calibration

[teacher_reference.portable.json](teacher_reference.portable.json) contains the
existing calibration at 20 episodes per task. Task statistics, training
semantics, and original provenance are retained; checkpoint paths are portable
placeholders. Verify model contents before generating a machine-local copy:

```bash
export OPD_STUDENT_CHECKPOINT=/absolute/path/C10/checkpoint-60000
python -m toolkits.robocasa_mopd.resources prepare \
  --output /absolute/path/inputs/teacher_reference.json
export OPD_TEACHER_REFERENCE=/absolute/path/inputs/teacher_reference.json
```

The script reads `OPD_MODEL_ROOT`, checks all four models, and binds teacher
paths only after verification. Existing output files are not overwritten.
Do not manually bind unrelated teachers to an old reference. Calibration used
1,000 episodes and 1,299,775 environment control steps. Account for it separately;
it is not the final independent evaluation. M0/M1 do not use its success values;
M2/M3 share the frozen reference. Increasing student evaluation to 50 episodes
per task does not change this calibration.

## Simulator assets

The profile does not include kitchen, object, or texture assets. Use the pinned
RoboCasa checkout's scripts:

```bash
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

Alternatively, copy the agreed asset snapshot and configure its paths through
RoboCasa macros. The installer does not bind another server's shared disk.
Record asset provenance and compare file manifests when reusing a snapshot.
A complete asset hash manifest is not currently published in Git, so the
repository does not claim byte-identical simulator inputs. Fix missing assets
rather than skipping tasks; receiving-machine training and evaluation smoke
tests remain required.
