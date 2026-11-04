#!/usr/bin/env bash
#SBATCH --job-name=robotwin-formal
#SBATCH --partition=hpg-rtx6e0e
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
if [[ $# -ne 2 || ( "$2" != validate && "$2" != run ) ]]; then
    echo "Usage: sbatch toolkits/robotwin_mopd/formal/slurm.sh /absolute/machine.json {validate|run}" >&2
    exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must supply the allocated GPU mask}"
# One launcher owns its Ray cluster. Never spawn four copies of the launcher.
entry=toolkits/robotwin_mopd/formal/experiment.py
python3 "$entry" --machine "$1" prepare
python3 "$entry" --machine "$1" preflight
if [[ "$2" == validate ]]; then
    python3 "$entry" --machine "$1" collective-smoke
    python3 "$entry" --machine "$1" teacher-smoke
    python3 "$entry" --machine "$1" smoke m1
    exit 0
fi
# Formal execution is a separate submission after reviewing deployment logs.
# Refuse missing/failed full-model smoke; never rerun and overwrite it.
python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path
machine = json.loads(Path(sys.argv[1]).read_text())
output = Path(machine["output_root"])
smoke = output / "smoke/robotwin_formal_sft2000_m1_rounds20_seed0_smoke"
if (smoke / "status").read_text().strip() != "COMPLETE":
    raise SystemExit("Complete and inspect validation before formal execution")
weights = smoke / "checkpoints/global_step_2/actor/model_state_dict/full_weights.pt"
if not weights.is_file() or not weights.stat().st_size:
    raise SystemExit("Missing full-model smoke checkpoint")
for rank in range(4):
    if json.loads((output / f"smoke/tiny_fsdp/rank{rank}.json").read_text())["status"] != "PASS":
        raise SystemExit("Distributed smoke incomplete")
for task in ("adjust_bottle", "handover_mic", "place_container_plate", "stack_bowls_two"):
    if json.loads((output / f"smoke/teachers/{task}.json").read_text())["completed_episodes"] != 2:
        raise SystemExit("Teacher smoke incomplete")
PY
python3 "$entry" --machine "$1" eval teachers
python3 "$entry" --machine "$1" eval student
for method in m0 m1 m2 m3; do
    python3 "$entry" --machine "$1" train "$method"
    python3 "$entry" --machine "$1" eval "$method"
done
python3 "$entry" --machine "$1" summarize
