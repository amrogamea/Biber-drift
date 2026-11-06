#!/bin/bash
# Paraphrase the sample with open-weight LLMs: one array task per model, one GPU each (vLLM).
# Set up for NHR@FAU TinyGPU (login node tinyx), submitted with its wrapper sbatch.tinygpu:
#
#   mkdir -p logs
#   sbatch.tinygpu hpc/generate.sbatch pilot             # the pilot subset (400 texts), the six main models
#   sbatch.tinygpu hpc/generate.sbatch main              # the full sample (15,357 texts)
#   sbatch.tinygpu --array=6 hpc/generate.sbatch main    # the Llama 2 check (7th model in the config)
#   sbatch.tinygpu --array=0-6 --time=00:30:00 hpc/generate.sbatch pilot --limit 20 --name smoke
#
# Arguments after the stage go to `biber-drift paraphrase generate` (--limit, --name, --set).
# Command-line options override the lines below, e.g. --array=0-6 or --time=01:00:00 for the pilot
# (a short limit starts sooner on a busy queue). Array index i runs the i-th model of
# configs/paraphrase.yaml. A task that stops or runs out of time resumes where it ended when
# submitted again. Set HF_HOME (the model cache) and VLLM_ENV before submitting.
#
# TinyGPU's partition a100 has 4x A100 40 GB per node; every GPU comes with 32 cores and 117 GB of
# RAM, so the job asks for no cores or memory. The account is the default one (iwso). The other
# partitions do not fit: RTX 3080 / 2080 Ti have 10-11 GB, and vLLM does not run on V100s.
#SBATCH --job-name=paraphrase
#SBATCH --array=0-5
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.log
set -euo pipefail

STAGE="${1:-pilot}"
shift $(( $# > 0 ? 1 : 0 ))   # the rest goes to `paraphrase generate`, e.g. --limit 20 --name smoke
VLLM_ENV="${VLLM_ENV:-$HOME/envs/vllm}"
export HF_HOME="${HF_HOME:?set HF_HOME to the model cache on project storage before submitting}"
export HF_HUB_OFFLINE=1   # load the cached weights only; compute nodes may have no internet

cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck disable=SC1091
source "$VLLM_ENV/bin/activate"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
biber-drift paraphrase generate --config configs/paraphrase.yaml --stage "$STAGE" \
  --model "${SLURM_ARRAY_TASK_ID:?submit with sbatch so the array index picks the model}" "$@"
