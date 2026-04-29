#!/usr/bin/env python3
"""
Fluxway Release & Deployment Script
-----------------------------------
A standard release utility for building and publishing the Fluxway package.
Supports multiple publishing targets including PyPI, AWS S3, and Minio.
"""

import abc
import argparse
import os
import subprocess
import sys
from pathlib import Path


class WheelBuilder:
    """Handles the building of the PEP-517 compliant wheel."""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.dist_dir = self.project_dir / "dist"

    def build(self):
        print("=> [Builder] Starting wheel build using standard `build` module...")
        try:
            subprocess.run(
                [sys.executable, "-m", "build", "--wheel"],
                cwd=self.project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error: Build failed: {e.stderr}", file=sys.stderr)
            raise RuntimeError("Build process encountered an error") from e

    def get_latest_wheel(self) -> Path:
        if not self.dist_dir.exists():
            raise FileNotFoundError(f"Dist directory not found: {self.dist_dir}")

        wheels = list(self.dist_dir.glob("*.whl"))
        if not wheels:
            raise FileNotFoundError(f"No wheels found in {self.dist_dir}")

        return max(wheels, key=lambda p: p.stat().st_mtime)


class BaseUploader(abc.ABC):
    """Abstract base class defining the uploader interface."""

    @abc.abstractmethod
    def upload(self, file_path: Path, dry_run: bool = False):
        pass


class TwineUploader(BaseUploader):
    """Publishes the wheel to PyPI via Twine."""

    def __init__(self, repository: str = "pypi"):
        self.repository = repository

    def upload(self, file_path: Path, dry_run: bool = False):
        cmd = [
            sys.executable,
            "-m",
            "twine",
            "upload",
            "--repository",
            self.repository,
            str(file_path),
        ]
        print(f"=> [TwineUploader] Publishing to {self.repository}: {' '.join(cmd)}")
        if dry_run:
            print("=> [Dry Run] Skipped actual upload.")
            return

        try:
            subprocess.run(cmd, check=True)
            print("=> [TwineUploader] Upload successful!")
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Twine upload failed.") from e


class S3Uploader(BaseUploader):
    """Uploads the wheel to an AWS S3 bucket."""

    def __init__(self, s3_uri: str):
        if not s3_uri.startswith("s3://"):
            raise ValueError("S3 URI must start with 's3://'")
        self.s3_uri = s3_uri

    def upload(self, file_path: Path, dry_run: bool = False):
        cmd = ["aws", "s3", "cp", str(file_path), self.s3_uri]
        print(f"=> [S3Uploader] Copying to S3: {' '.join(cmd)}")
        if dry_run:
            print("=> [Dry Run] Skipped actual upload.")
            return

        try:
            subprocess.run(cmd, check=True)
            print("=> [S3Uploader] Upload successful!")
        except subprocess.CalledProcessError as e:
            raise RuntimeError("S3 upload failed.") from e


class MinioUploader(BaseUploader):
    """Uploads the wheel to a Minio instance."""

    def __init__(self, target_path: str):
        self.target_path = target_path

    def upload(self, file_path: Path, dry_run: bool = False):
        cmd = ["mc", "cp", str(file_path), self.target_path]
        print(f"=> [MinioUploader] Copying to Minio: {' '.join(cmd)}")
        if dry_run:
            print("=> [Dry Run] Skipped actual upload.")
            return

        try:
            subprocess.run(cmd, check=True)
            print("=> [MinioUploader] Upload successful!")
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Minio upload failed.") from e


class UploaderFactory:
    """Factory to create appropriate uploader based on user intent."""

    @staticmethod
    def create(backend: str, target: str) -> BaseUploader:
        if backend == "pypi":
            return TwineUploader(repository=target or "pypi")
        elif backend == "s3":
            if not target:
                raise ValueError("Target S3 URI must be provided for backend 's3'")
            return S3Uploader(s3_uri=target)
        elif backend == "minio":
            if not target:
                raise ValueError("Target path must be provided for backend 'minio'")
            return MinioUploader(target_path=target)
        else:
            raise ValueError(f"Unsupported backend: {backend}")


def main():
    parser = argparse.ArgumentParser(
        description="Build and publish Fluxway wheels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["pypi", "s3", "minio"],
        default="pypi",
        help="The deployment backend to use.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="",
        help="Target destination (e.g. 's3://bucket/path/', 'minio1/public/wheels/', or PyPI repo name). Required for s3 and minio.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the wheel and print the upload command without executing it.",
    )
    args = parser.parse_args()

    # Determine target based on environment if not provided
    target = args.target
    if not target:
        if args.backend == "minio":
            target = os.environ.get(
                "FLUXWAY_MINIO_TARGET", "minio1/public/wheels/fluxway/"
            )
        elif args.backend == "s3":
            target = os.environ.get("FLUXWAY_S3_TARGET", "")

    try:
        uploader = UploaderFactory.create(args.backend, target)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    project_dir = Path(__file__).parent.parent.absolute()

    try:
        builder = WheelBuilder(project_dir)
        builder.build()
        latest_wheel = builder.get_latest_wheel()
        print(f"=> [System] Found newly built wheel: {latest_wheel.name}")

        uploader.upload(latest_wheel, dry_run=args.dry_run)
    except Exception as e:
        print(f"Deployment aborted: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
