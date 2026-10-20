import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def release_workflow():
    content = (PROJECT_ROOT / ".github" / "workflows" / "release.yml").read_text()
    return yaml.load(content, Loader=yaml.BaseLoader)


def _run_step(workflow, job, name, directory, env):
    step = next(
        item for item in workflow["jobs"][job]["steps"] if item.get("name") == name
    )
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("version", "tag", "prerelease"),
    [
        ("1.2.3", "v1.2.3", "false"),
        ("1.2.3rc1", "v1.2.3rc1", "true"),
        ("1.2.3", "v1.2.4", None),
        ("0+unreleased", "v0+unreleased", None),
    ],
)
def test_release_version_step_matches_tag_and_writes_github_outputs(
    release_workflow, tmp_path, version, tag, prerelease
):
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(
        PROJECT_ROOT / "scripts" / "distribution.py",
        tmp_path / "scripts" / "distribution.py",
    )
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
    output = tmp_path / "github-output"
    env = dict(
        os.environ,
        PATH=os.pathsep.join((str(Path(sys.executable).parent), os.environ["PATH"])),
        RELEASE_TAG=tag,
        GITHUB_OUTPUT=str(output),
    )
    result = _run_step(
        release_workflow, "validate", "Match tag to project version", tmp_path, env
    )
    if prerelease is None:
        assert result.returncode != 0
        assert not output.read_text()
    else:
        assert result.returncode == 0, result.stderr
        assert dict(line.split("=", 1) for line in output.read_text().splitlines()) == {
            "version": version,
            "prerelease": prerelease,
        }


@pytest.mark.parametrize("relationship", ["main", "ancestor", "feature"])
def test_release_main_gate_rejects_unmerged_commits(
    release_workflow, tmp_path, relationship
):
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="Release Gate Test",
        GIT_AUTHOR_EMAIL="release-test@example.invalid",
        GIT_COMMITTER_NAME="Release Gate Test",
        GIT_COMMITTER_EMAIL="release-test@example.invalid",
    )
    for variable in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(variable, None)

    def git(*args, content=None):
        result = subprocess.run(
            ["git", "-c", "commit.gpgsign=false", *args],
            cwd=tmp_path,
            env=env,
            input=content,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()

    git("init", "--initial-branch=main")
    tree = git("mktree", content="")
    main = git("commit-tree", tree, content="main fixture\n")
    child = git("commit-tree", tree, "-p", main, content="child fixture\n")
    git(
        "update-ref",
        "refs/remotes/origin/main",
        child if relationship == "ancestor" else main,
    )
    git(
        "update-ref", "--no-deref", "HEAD", child if relationship == "feature" else main
    )
    result = _run_step(
        release_workflow, "validate", "Require a commit on main", tmp_path, env
    )
    assert (result.returncode == 0) == (relationship != "feature"), result.stderr


@pytest.mark.parametrize("prerelease", [False, True])
def test_github_release_step_attaches_artifacts_and_sets_release_kind(
    release_workflow, tmp_path, prerelease
):
    executable = tmp_path / "bin"
    executable.mkdir()
    gh = executable / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE_GH_ARGS']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    gh.chmod(0o755)
    dist = tmp_path / "dist"
    dist.mkdir()
    filenames = ("fluxway-1.2.3-py3-none-any.whl", "fluxway-1.2.3.tar.gz")
    for filename in filenames:
        (dist / filename).touch()
    output = tmp_path / "arguments.json"
    tag = "v1.2.3rc1" if prerelease else "v1.2.3"
    env = dict(
        os.environ,
        PATH=os.pathsep.join((str(executable), os.environ["PATH"])),
        RELEASE_TAG=tag,
        RELEASE_PRERELEASE=str(prerelease).lower(),
        CAPTURE_GH_ARGS=str(output),
    )
    result = _run_step(
        release_workflow,
        "github-release",
        "Attach the published distributions",
        tmp_path,
        env,
    )
    assert result.returncode == 0, result.stderr
    arguments = json.loads(output.read_text())
    assert arguments == [
        "release",
        "create",
        tag,
        *(f"dist/{filename}" for filename in filenames),
        "--verify-tag",
        "--generate-notes",
        "--title",
        tag,
        *(["--prerelease", "--latest=false"] if prerelease else []),
    ]
