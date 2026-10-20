# Contributing to Fluxway

[中文版](CONTRIBUTING_zh.md)

Thank you for contributing to Fluxway! This document outlines local development setup, code formatting, running tests, and opening pull requests.

---

## 1. Prerequisites & Environment Setup

* **OS & Python**: Linux, **Python 3.12 or 3.13**.
* Create a dedicated virtual environment and install development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[jax,profiler,visualization]' --group test --group lint --group build
```

---

## 2. Local Quality & Pre-commit Checks

Before pushing or opening a PR, ensure all local linters, unit tests, and package builds pass cleanly (matching our GitHub CI pipeline):

### Step A: Lint & Code Formatting
```bash
bash scripts/format.sh --check
# Apply lint fixes and formatting:
bash scripts/format.sh
```

### Step B: Run Unit Tests (CPU)
```bash
JAX_PLATFORMS=cpu python -m pytest -ra --strict-markers --strict-config \
  --ignore=tests/checkpointer/test_retention.py \
  --ignore=tests/example/test_sparse_checkpoint_e2e.py
```
> *Note*: The two ignored test files require proprietary native `sparse_embedding` libraries and Lance; standard portable CI runs on CPU without them.

### Step C: Build & Validate Distribution Packages
```bash
# Clean previous build artifacts
rm -rf dist/
python -m build
python -m twine check --strict dist/*
python scripts/distribution.py check dist

# Optional: Validate clean wheel installation in an isolated sandbox
python -m venv tmp/wheel-env
tmp/wheel-env/bin/python -m pip install dist/*.whl
tmp/wheel-env/bin/python -m pip check
tmp/wheel-env/bin/python -I scripts/check_installed.py
```

---

## 3. Pull Request Guidelines

1. **Focused Scope**: Each PR should address a single feature or bug fix. Avoid mixing unrelated changes.
2. **Include Tests & Docs**: Any behavior change or bug fix must come with corresponding unit tests and documentation updates (remember to update both English and Chinese docs).
3. **No Breaking Surprises**: Clearly document any breaking API modifications and migrations in the PR description.
4. **Clean CI**: All GitHub Actions workflows must pass on the final commit before merging.

For versioning and publishing procedures, please see [Releasing Guide](docs/releasing.md) ([中文](docs/releasing_zh.md)).

The scheduled [Nightly wheel](.github/workflows/nightly.yml) workflow reuses the
complete CI pipeline and stores its verified wheel and source distribution in the
`python-distributions` Actions artifact for 14 days. You can dispatch it manually
after the workflow is present on `main`. Changes to the CI or nightly workflow
also run nightly validation on the pull request before merging.
