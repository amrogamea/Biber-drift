#!/usr/bin/env bash
# Run the biber-drift pipeline end to end.
#
#   1. Put your data under data/raw/ (see data/README.md), or point DATA_PATH at it.
#   2. Adjust the settings below -- or pass them as environment variables:
#        DATASET=acl ./run.sh
#   3. Run it:
#        ./run.sh                                   # all stages
#        ./run.sh --stages analyze,report           # extra arguments go straight to the CLI
#        ./run.sh --set mmd.sample_size=300         # override any config value
#
# The first run creates a virtual environment, installs the package and downloads the spaCy
# model; later runs reuse them.
set -euo pipefail

# =============================== settings =========================================
DATASET="${DATASET:-ao3}"                      # ao3 | acl  -> uses configs/<DATASET>.yaml
CONFIG="${CONFIG:-configs/${DATASET}.yaml}"    # or point to your own config file
DATA_PATH="${DATA_PATH:-}"                     # empty = data.path from the config
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"            # results go to $OUTPUT_DIR/<run name>/
RUN_NAME="${RUN_NAME:-}"                       # empty = name from the config
STAGES="${STAGES:-prepare,features,analyze,report}"
SPACY_MODEL="${SPACY_MODEL:-}"                 # empty = features.spacy_model from the config
OVERRIDES=(                                    # extra config overrides, one per line, e.g.
  # "mmd.sample_size=300"
  # "mmd.bootstrap_resamples=1000"
)
PYTHON="${PYTHON:-}"                           # empty = newest python3.x >= 3.10 on the PATH
VENV_DIR="${VENV_DIR:-.venv}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"              # 1 = use the active environment as-is
# ===================================================================================

cd "$(dirname "$0")"

# Load secrets such as S2_API_KEY (only needed for `biber-drift acquire-acl`).
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG" >&2
  exit 1
fi

python_ok() {  # is "$1" a Python >= 3.10?
  "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null
}

if [[ "$SKIP_INSTALL" != "1" ]]; then
  if [[ -z "$PYTHON" ]]; then
    for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
      if command -v "$candidate" >/dev/null 2>&1 && python_ok "$candidate"; then
        PYTHON="$candidate"
        break
      fi
    done
    if [[ -z "$PYTHON" ]]; then
      echo "Python 3.10 or newer not found on the PATH; install it or set PYTHON=/path/to/python3.x" >&2
      exit 1
    fi
  elif ! python_ok "$PYTHON"; then
    echo "PYTHON=$PYTHON is older than 3.10; point it to Python 3.10 or newer" >&2
    exit 1
  fi
  # A venv made earlier with a too-old Python (e.g. macOS's built-in 3.9) is rebuilt.
  if [[ -f "$VENV_DIR/pyvenv.cfg" ]] && ! python_ok "$VENV_DIR/bin/python"; then
    echo ">> $VENV_DIR uses a Python older than 3.10; recreating it with $PYTHON"
    rm -rf "$VENV_DIR"
  fi
  # A venv is tied to the folder it was created in; after the project is moved it is rebuilt.
  if [[ -f "$VENV_DIR/bin/activate" ]] && \
     ! grep -qF "$(cd "$VENV_DIR" && pwd -P)" "$VENV_DIR/bin/activate"; then
    echo ">> $VENV_DIR was created in another folder; recreating it"
    rm -rf "$VENV_DIR"
  fi
  if [[ ! -d "$VENV_DIR" ]]; then
    echo ">> Creating virtual environment in $VENV_DIR with $PYTHON ($("$PYTHON" --version 2>&1))"
    "$PYTHON" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  stamp="$VENV_DIR/.biber-drift-installed"
  if [[ ! -f "$stamp" || pyproject.toml -nt "$stamp" ]]; then
    echo ">> Installing biber-drift and its dependencies"
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet -e ".[dev]"
    touch "$stamp"
  fi
fi

if [[ -z "$SPACY_MODEL" ]]; then
  SPACY_MODEL="$(python -c "from biber_drift.config import load_config; print(load_config('$CONFIG').features.spacy_model)")"
fi
if ! python -c "import spacy, sys; sys.exit(0 if spacy.util.is_package('$SPACY_MODEL') else 1)"; then
  echo ">> Downloading spaCy model $SPACY_MODEL"
  python -m spacy download "$SPACY_MODEL"
fi

args=(run --config "$CONFIG" --stages "$STAGES" --output-dir "$OUTPUT_DIR"
      --set "features.spacy_model=$SPACY_MODEL")
if [[ -n "$DATA_PATH" ]]; then args+=(--data-path "$DATA_PATH"); fi
if [[ -n "$RUN_NAME" ]]; then args+=(--name "$RUN_NAME"); fi
# ${array[@]+...} keeps `set -u` happy with an empty array on bash 3.2 (macOS).
for override in ${OVERRIDES[@]+"${OVERRIDES[@]}"}; do
  args+=(--set "$override")
done

echo ">> biber-drift ${args[*]} $*"
python -m biber_drift "${args[@]}" "$@"
