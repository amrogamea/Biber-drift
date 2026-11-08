#!/usr/bin/env bash
# Create the tagging environment for `biber-drift paraphrase tag`: spaCy en_core_web_trf (on a CUDA
# GPU when there is one) and pybiber. Run once on the HPC login node (it needs internet), from the
# project folder. spaCy, the model, spacy-curated-transformers and pybiber are pinned to the
# laptop's versions, which tagged the pipeline's features; the originals are re-tagged here anyway,
# so that the remaining differences (torch, CPU vs GPU) cannot pose as change.
#
#   PYTHON=/apps/python/3.12-conda/bin/python3.12 bash hpc/setup_tag_env.sh
#   TAG_ENV=/project/me/envs/tag bash hpc/setup_tag_env.sh
set -euo pipefail

TAG_ENV="${TAG_ENV:-$HOME/envs/tag}"
PYTHON="${PYTHON:-python3.12}"
TRF_WHEEL="https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl"

cd "$(dirname "$0")/.."
"$PYTHON" -m venv "$TAG_ENV"
# shellcheck disable=SC1091
source "$TAG_ENV/bin/activate"
python -m pip install --upgrade pip
# One resolver run for everything. Not spacy[cuda12x]: it pins cupy 12, which needs numpy < 2,
# while torch from PyPI brings CUDA 13 libraries, which cupy-cuda13x uses. polars[rtcompat]: the
# default polars build needs AVX2, which the TinyGPU login node (Xeon E5-2650 v2) lacks;
# polars < 2 because pybiber 0.3.1 calls str.concat, removed in polars 2.0.
python -m pip install "spacy==3.8.16" "spacy-curated-transformers==0.3.1" "pybiber==0.3.1" "$TRF_WHEEL" \
  "torch==2.15.0" "cupy-cuda13x==14.2.0" "polars[rtcompat]==1.44.2" -e ".[dev]"
python - <<'EOF'
import importlib.metadata as md
import polars, pybiber, spacy, torch
print("spacy", spacy.__version__, "| en_core_web_trf", md.version("en_core_web_trf"),
      "| spacy-curated-transformers", md.version("spacy-curated-transformers"),
      "| pybiber", md.version("pybiber"), "| polars", polars.__version__, "| numpy", md.version("numpy"),
      "| cupy", md.version("cupy-cuda13x"), "| torch", torch.__version__, "| CUDA", torch.version.cuda)
EOF
echo ">> Done. Activate with: source $TAG_ENV/bin/activate"
