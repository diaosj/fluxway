from importlib.metadata import distribution
from pathlib import Path
import sys

import numpy as np
from numpy.typing import NDArray

import fluxway
from fluxway.common.types.tensor import TransportTensor


def main() -> None:
    eager_backends = {"jax", "orbax", "etils"}.intersection(sys.modules)
    if eager_backends:
        raise RuntimeError(f"Core import loaded optional backends: {eager_backends}")
    package = Path(fluxway.__file__).resolve().parent
    installed = distribution("fluxway")
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(
            f"Imported Fluxway outside the installation environment: {package}"
        )
    if package != Path(str(installed.locate_file("fluxway"))).resolve():
        raise RuntimeError("Imported Fluxway does not match its distribution metadata")
    if not (package / "py.typed").is_file():
        raise RuntimeError("Installed Fluxway is missing its typing marker")
    source: NDArray[np.int64] = np.arange(8, dtype=np.int64)
    tensor = TransportTensor(source)
    np.testing.assert_array_equal(tensor.flat_values, source)
    print(f"Installed fluxway {installed.version}: {package}")


if __name__ == "__main__":
    main()
