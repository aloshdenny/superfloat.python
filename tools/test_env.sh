#!/bin/bash
# Build the pinned DSL test environment (env/dsl-test.txt) in a fresh venv and print its versions.
#   tools/test_env.sh [venv dir, default .venv-dsl] [python, default python3.10]
# Then: <venv>/bin/python -m pytest -q -n auto
set -euo pipefail
cd "$(dirname "$0")/.."
V=${1:-.venv-dsl}; PY=${2:-python3.10}
rm -rf "$V"
"$PY" -m venv "$V"
"$V/bin/pip" install -q --upgrade pip
"$V/bin/pip" install -q -e ".[test,hf,xdist]" -c env/dsl-test.txt --extra-index-url https://download.pytorch.org/whl/cpu
"$V/bin/python" -c "import sys, numpy, torch, transformers; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
