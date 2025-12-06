#!/bin/bash

set -euf -o pipefail  # Stop at failure.

# python3 -m venv .venv
# . .venv/bin/activate
# pip install -U pip
# pip install -r requirements.txt

export PYTHONPATH=$(pwd)/..
echo $PYTHONPATH
python3 main.py "$@"
