#!/bin/bash

# Format the codebase using ruff
ruff format fluxway/ tests/ scripts/ setup.py
ruff check fluxway/ tests/ scripts/ setup.py --fix
