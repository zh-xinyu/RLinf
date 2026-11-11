#!/usr/bin/env bash
#SBATCH --job-name=robotwin-new4
#SBATCH --partition=hpg-rtx6e0e
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
[[ $# == 2 && ( "$2" == validate || "$2" == run ) ]] || { echo "Usage: sbatch slurm.sh /absolute/machine.json {validate|run}" >&2; exit 2; }
cd "${SLURM_SUBMIT_DIR:?Submit from the release repository root}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must provide exactly four allocated GPU IDs}"
exec python3 -m toolkits.robotwin_mopd.new4.experiment --machine "$1" "$2"
