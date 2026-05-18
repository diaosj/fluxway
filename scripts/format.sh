#!/usr/bin/env bash
set -euo pipefail

# Get the directory of the script and change to the project root
cd "$(dirname "$0")/.."

# Resolve ruff: prefer python -m ruff for virtualenv compatibility
if command -v ruff &>/dev/null; then
    RUFF="ruff"
else
    RUFF="python -m ruff"
fi

# Format the codebase using ruff
$RUFF format fluxway/ tests/ setup.py
$RUFF check fluxway/ tests/ setup.py --fix
