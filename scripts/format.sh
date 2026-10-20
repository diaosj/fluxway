#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

lint_args=(--fix)
format_args=()
if (( $# )); then
    if [[ $# -ne 1 || "$1" != --check ]]; then
        printf 'Usage: bash scripts/format.sh [--check]\n' >&2
        exit 2
    fi
    lint_args=()
    format_args=(--check)
fi

python -m ruff check . "${lint_args[@]}"
python -m ruff format . "${format_args[@]}"
