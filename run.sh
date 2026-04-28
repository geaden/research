#!/bin/bash

set -euf -o pipefail  # Stop at failure.

# python3 -m venv .venv
# . .venv/bin/activate
# pip install -U pip
# pip install -r requirements/testing.txt

WD=$1
shift 1
export PYTHONPATH="$(pwd)"
pushd $WD
python3 main.py "$@"
popd
