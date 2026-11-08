#!/usr/bin/env bash
# Create the vLLM environment for the paraphrase step. Run once on the HPC login node (it needs
# internet), from the project folder. vLLM needs Python 3.11-3.14: load one first if the HPC uses
# modules (e.g. `module load python/3.12`).
#
#   bash hpc/setup_vllm_env.sh
#   VLLM_ENV=/project/me/envs/vllm PYTHON=python3.11 bash hpc/setup_vllm_env.sh
set -euo pipefail

VLLM_ENV="${VLLM_ENV:-$HOME/envs/vllm}"
VLLM_VERSION="${VLLM_VERSION:-0.32.0}"   # latest on PyPI on 2026-10-29; each output's .meta.json records it
PYTHON="${PYTHON:-python3.12}"

cd "$(dirname "$0")/.."
"$PYTHON" -m venv "$VLLM_ENV"
# shellcheck disable=SC1091
source "$VLLM_ENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install "vllm==$VLLM_VERSION" pandas pyarrow beautifulsoup4
# The project without its analysis dependencies (spaCy and pybiber belong in the tagging
# environment). beautifulsoup4 is only there because the command-line tool imports the AO3 scraper.
python -m pip install --no-deps -e .
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| CUDA', torch.version.cuda)"
echo ">> Done. Activate with: source $VLLM_ENV/bin/activate"
