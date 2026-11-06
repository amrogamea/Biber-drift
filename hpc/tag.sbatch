#!/bin/bash
# Clean, filter and Biber-tag the outputs of one generation run, and re-tag the sample's originals,
# with spaCy en_core_web_trf on a GPU (`biber-drift paraphrase tag`). One array task per key:
# task 0 tags the originals, task i >= 1 the model at index i-1 of configs/paraphrase.yaml.
# Set up for NHR@FAU TinyGPU; the tagging environment comes from hpc/setup_tag_env.sh:
#
#   mkdir -p logs
#   sbatch.tinygpu --export=ALL hpc/tag.sbatch main          # the originals and all seven models
#   sbatch.tinygpu --export=ALL --array=3 hpc/tag.sbatch main  # one model again (index 2 + 1)
#
# Arguments after the stage go to `biber-drift paraphrase tag` (--name, --limit, --set). A task that
# stops resumes from its tagging cache when submitted again; a finished key is skipped. Set TAG_ENV
# before submitting. One RTX 3080 (10 GB) is enough for en_core_web_trf.
#SBATCH --job-name=tag
#SBATCH --array=0-7
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.log
set -euo pipefail

STAGE="${1:-main}"
shift $(( $# > 0 ? 1 : 0 ))
TAG_ENV="${TAG_ENV:-$HOME/envs/tag}"
TASK="${SLURM_ARRAY_TASK_ID:?submit with sbatch so the array index picks the key}"
if [ "$TASK" -eq 0 ]; then KEY=original; else KEY=$((TASK - 1)); fi

cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck disable=SC1091
source "$TAG_ENV/bin/activate"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# require: fail at once if spaCy cannot use the GPU, instead of tagging for hours on the CPU
biber-drift paraphrase tag --config configs/paraphrase.yaml --stage "$STAGE" --model "$KEY" \
  --set tagging.gpu=require "$@"
