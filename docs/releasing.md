# Releasing Fluxway

> **Quick Summary**
>
> * **Single Source of Truth**: The release version is strictly determined by `project.version` in `pyproject.toml`. The default `0+unreleased` is for local development and will be rejected by publishing workflows.
> * **Release Trigger**: Pushing a `v*` tag triggers GitHub Actions to run full CI, build sdist/wheel, publish to PyPI, and draft/publish a GitHub Release.
> * **Zero Token Secrets**: Releases use PyPI Trusted Publishing (OIDC). No long-lived API tokens or repo secrets are needed.

---

## 1. Initial Setup (One-time)

1. In GitHub repository settings, create an Environment named `pypi`, allowing deployment only from tags matching `v*`.
2. In PyPI project settings (or register as a pending publisher if the project is not yet on PyPI), add a **GitHub Trusted Publisher**:
   * **PyPI project**: `fluxway`
   * **GitHub owner**: `fluxway-project`
   * **Repository**: `fluxway`
   * **Workflow filename**: `release.yml`
   * **Environment**: `pypi`

*(No API tokens or secrets required. Ref: [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/))*

---

## 2. Release Steps

1. **Bump Version (via PR)**:
   * Create a pull request updating `project.version` in `pyproject.toml` to a canonical PEP 440 version (e.g. `0.1.0` or `0.1.0rc1`).
   * Detail any breaking changes and migration guides in the PR description.
2. **Merge PR**:
   * Once CI passes and reviews are complete, merge into `main`.
3. **Tag & Push**:
   * On the fresh merge commit on `main`, run local pre-checks, tag, and push:
     ```bash
     git checkout main && git pull
     python scripts/distribution.py release v<version>
     git tag -a v<version> -m "Release v<version>"
     git push origin v<version>
     ```
4. **Automated Pipeline**:
   * The `Release` GitHub Action automatically:
     1. Validates that the tag matches `pyproject.toml` and belongs to `main`.
     2. Runs complete CI (linting, test matrix across Python versions, package integrity checks).
     3. Builds sdist and wheel, and tests installation in an isolated clean environment.
     4. Publishes distributions to PyPI via OIDC.
     5. Creates a GitHub Release with build artifacts and auto-generated release notes.

---

## 3. What CI & Build Validates

* **Lint & Format**: Zero-tolerance Ruff linting and formatting.
* **Portability Tests**: Python 3.12 & 3.13 matrix covering JAX, Orbax, Grain, Flax, SafeTensors, profiler, and visualizers.
* **Packaging Integrity**: Strict Twine checks, `py.typed` marker, license inclusion, and isolated import verification.
* **Modular Dependencies**: Core wheel functions without heavy optional backends; missing backends provide explicit installation guidance rather than crashing unexpectedly.

---

## 4. Troubleshooting & Failed Releases

* **No Overwrites or Retags**: PyPI strictly forbids re-uploading an existing file version. If a release has already been partially uploaded or published, bump `project.version` and release a new patch version.
* **PyPI Upload Failures**: Inspect the workflow log. If it is an OIDC permission/configuration issue and no files were published to PyPI, correct the environment settings and rerun the failed job.
* **PyPI Succeeded but GitHub Release Failed**: Do not re-run the entire pipeline from scratch (PyPI will reject duplicate uploads). Manually attach the build artifacts from the workflow run to the GitHub Release, or rerun only the failed release step.
