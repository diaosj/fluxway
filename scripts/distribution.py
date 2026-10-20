import argparse
from email.parser import BytesParser
from pathlib import Path
import tarfile
import tomllib
from zipfile import ZipFile

from packaging.utils import (
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import Version


def release_metadata(version: str, tag: str) -> dict[str, str]:
    parsed = Version(version)
    if (
        version != str(parsed)
        or len(parsed.release) != 3
        or parsed.epoch
        or parsed.local is not None
        or parsed.is_devrelease
    ):
        raise ValueError(f"Release requires a canonical X.Y.Z version: {version!r}")
    if tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} does not match project version v{version}")
    return {"version": version, "prerelease": str(parsed.is_prerelease).lower()}


def _check_metadata(content: bytes, project: dict) -> None:
    metadata = BytesParser().parsebytes(content)
    if canonicalize_name(metadata["Name"]) != canonicalize_name(project["name"]):
        raise ValueError(f"Distribution name does not match {project['name']!r}")
    for field, key in (("Version", "version"), ("Requires-Python", "requires-python")):
        if metadata[field] != project[key]:
            raise ValueError(f"Distribution {field} does not match {project[key]!r}")
    if metadata["License-Expression"] != project["license"]:
        raise ValueError("Distribution license does not match project metadata")


def check_distributions(directory: Path, project_root: Path) -> None:
    project = tomllib.loads((project_root / "pyproject.toml").read_text())["project"]
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1 or len(list(directory.iterdir())) != 2:
        raise ValueError("Expected exactly one wheel and one source distribution")
    expected_identity = (
        canonicalize_name(project["name"]),
        Version(project["version"]),
    )
    if parse_wheel_filename(wheels[0].name)[:2] != expected_identity:
        raise ValueError("Wheel filename does not match project name and version")
    if parse_sdist_filename(sdists[0].name) != expected_identity:
        raise ValueError("Source distribution filename does not match project metadata")

    package_files = {
        path.relative_to(project_root).as_posix()
        for path in (project_root / "fluxway").rglob("*.py")
    } | {"fluxway/py.typed"}
    with ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
        if not package_files <= names:
            raise ValueError(f"Wheel is missing package files: {package_files - names}")
        if any(name.startswith(("tests/", "tmp/", "scripts/")) for name in names):
            raise ValueError("Wheel contains files outside the runtime package")
        metadata_files = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError("Wheel must contain exactly one METADATA file")
        _check_metadata(wheel.read(metadata_files[0]), project)
        license_path = metadata_files[0].removesuffix("METADATA") + "licenses/LICENSE"
        if wheel.read(license_path) != (project_root / "LICENSE").read_bytes():
            raise ValueError("Wheel license does not match the project license")

    with tarfile.open(sdists[0]) as sdist:
        prefix = sdists[0].name.removesuffix(".tar.gz") + "/"
        names = {name.removeprefix(prefix) for name in sdist.getnames()}
        required = package_files | {"pyproject.toml", "LICENSE", "PKG-INFO"}
        if not required <= names:
            raise ValueError(
                f"Source distribution is missing files: {required - names}"
            )
        metadata_file = sdist.extractfile(prefix + "PKG-INFO")
        if metadata_file is None:
            raise ValueError("Source distribution PKG-INFO must be a regular file")
        _check_metadata(metadata_file.read(), project)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("directory", type=Path)
    release = commands.add_parser("release")
    release.add_argument("tag")
    args = parser.parse_args()
    if args.command == "check":
        check_distributions(args.directory, args.project_root)
    else:
        project = tomllib.loads((args.project_root / "pyproject.toml").read_text())[
            "project"
        ]
        for key, value in release_metadata(project["version"], args.tag).items():
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
