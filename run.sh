#!/bin/bash

set -euf -o pipefail  # Stop at failure.

# python3 -m venv .venv
# . .venv/bin/activate
# pip install -U pip
# pip install -r requirements/testing.txt

WD=$1
shift 1
export PYTHONPATH="$(pwd)"
# Use project virtual environment if it exists.
PYTHON=python3
if [ -x "$(pwd)/.venv/bin/python3" ]; then
    PYTHON="$(pwd)/.venv/bin/python3"
fi
pushd $WD
$PYTHON main.py "$@"
popd
