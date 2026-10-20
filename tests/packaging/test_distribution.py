from io import BytesIO
import tarfile
from zipfile import ZipFile

import pytest

from scripts.distribution import check_distributions, release_metadata


@pytest.mark.parametrize(
    ("version", "prerelease"),
    [("1.2.3", "false"), ("1.2.3rc1", "true"), ("1.2.3a1", "true")],
)
def test_release_metadata_accepts_matching_canonical_versions(version, prerelease):
    assert release_metadata(version, f"v{version}") == {
        "version": version,
        "prerelease": prerelease,
    }


@pytest.mark.parametrize(
    "version",
    ["0+unreleased", "1.2.3.dev1", "1.2.3+local", "1.2", "1!1.2.3", "1.2.3-rc1"],
)
def test_release_metadata_rejects_nonrelease_versions(version):
    with pytest.raises(ValueError, match="canonical"):
        release_metadata(version, f"v{version}")


@pytest.mark.parametrize("tag", ["1.2.3", "v1.2.4", "v1.2.3\ncontents=write"])
def test_release_metadata_rejects_mismatched_tags(tag):
    with pytest.raises(ValueError, match="does not match"):
        release_metadata("1.2.3", tag)


@pytest.fixture
def distributions(tmp_path):
    (tmp_path / "fluxway").mkdir()
    (tmp_path / "fluxway" / "__init__.py").write_text("")
    (tmp_path / "fluxway" / "py.typed").touch()
    (tmp_path / "LICENSE").write_text("Apache license text")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "fluxway"\nversion = "1.2.3"\n'
        'requires-python = ">=3.12"\nlicense = "Apache-2.0"\n'
    )
    metadata = (
        b"Metadata-Version: 2.4\nName: fluxway\nVersion: 1.2.3\n"
        b"Requires-Python: >=3.12\nLicense-Expression: Apache-2.0\n"
    )
    wheel_files = {
        "fluxway/__init__.py": b"",
        "fluxway/py.typed": b"",
        "fluxway-1.2.3.dist-info/METADATA": metadata,
        "fluxway-1.2.3.dist-info/licenses/LICENSE": b"Apache license text",
    }
    sdist_files = {
        "fluxway/__init__.py": b"",
        "fluxway/py.typed": b"",
        "pyproject.toml": (tmp_path / "pyproject.toml").read_bytes(),
        "LICENSE": b"Apache license text",
        "PKG-INFO": metadata,
    }
    directory = tmp_path / "dist"
    directory.mkdir()

    def write(*, wheel_changes=None, sdist_changes=None):
        for files, changes in (
            (wheel_files, wheel_changes),
            (sdist_files, sdist_changes),
        ):
            for name, value in (changes or {}).items():
                if value is None:
                    del files[name]
                else:
                    files[name] = value
        with ZipFile(directory / "fluxway-1.2.3-py3-none-any.whl", "w") as wheel:
            for name, content in wheel_files.items():
                wheel.writestr(name, content)
        with tarfile.open(directory / "fluxway-1.2.3.tar.gz", "w:gz") as sdist:
            for name, content in sdist_files.items():
                info = tarfile.TarInfo(f"fluxway-1.2.3/{name}")
                info.size = len(content)
                sdist.addfile(info, BytesIO(content))
        return directory, tmp_path

    return write


def test_distribution_contains_the_complete_package(distributions):
    check_distributions(*distributions())


@pytest.mark.parametrize("name", ["fluxway/__init__.py", "fluxway/py.typed"])
def test_wheel_rejects_missing_runtime_files(distributions, name):
    with pytest.raises(ValueError, match="missing package files"):
        check_distributions(*distributions(wheel_changes={name: None}))


def test_wheel_rejects_stale_metadata(distributions):
    with pytest.raises(ValueError, match="Version does not match"):
        check_distributions(
            *distributions(
                wheel_changes={
                    "fluxway-1.2.3.dist-info/METADATA": (
                        b"Name: fluxway\nVersion: 1.2.2\nRequires-Python: >=3.12\n"
                    )
                }
            )
        )


def test_sdist_rejects_missing_source(distributions):
    with pytest.raises(ValueError, match="Source distribution is missing"):
        check_distributions(*distributions(sdist_changes={"fluxway/__init__.py": None}))


def test_wheel_rejects_nonruntime_files(distributions):
    with pytest.raises(ValueError, match="outside the runtime package"):
        check_distributions(*distributions(wheel_changes={"tmp/private.txt": b"data"}))


def test_distribution_rejects_extra_artifacts(distributions):
    directory, root = distributions()
    (directory / "old-release.whl").touch()
    with pytest.raises(ValueError, match="exactly one"):
        check_distributions(directory, root)
