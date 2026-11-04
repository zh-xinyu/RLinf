#!/usr/bin/env bash
#SBATCH --job-name=robocasa-mopd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm-%x-%j.out

# Submit from the repository root. Supply the confirmed partition/account on
# the sbatch command line. Slurm command-line resource options override defaults.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${SLURM_SUBMIT_DIR:?Missing Slurm submit directory}"
cd "$SLURM_SUBMIT_DIR"
if [ "$#" -lt 2 ]; then
    echo 'Usage: sbatch --partition=NAME [--account=NAME] toolkits/robocasa_mopd/slurm.sbatch /absolute/machine.json smoke|train|eval [arguments]' >&2
    exit 2
fi
MACHINE=$1
shift
test -f toolkits/robocasa_mopd/experiment.py
test -f "$MACHINE"
# An srun step supplies GPU visibility and CPU binding to Ray and its children.
# Do not export CUDA_VISIBLE_DEVICES or start Ray manually.
srun --ntasks=1 --cpu-bind=cores python3 toolkits/robocasa_mopd/experiment.py --machine "$MACHINE" "$@"
