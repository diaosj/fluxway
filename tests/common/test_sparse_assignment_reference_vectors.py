"""Backend-independent vectors for the sparse modulo runtime provider."""

import numpy as np
import pytest

from fluxway.common.mesh import DeviceMesh
from fluxway.common.tensor_backend import use_backend
from fluxway.common.types.sharding import IndexedModuloPartition

_REFERENCE_VECTORS = (
    (
        "int32",
        (-(2**31), -1, 0, 1, 2**31 - 1),
        (5, 6, 0, 1, 1),
    ),
    (
        "uint32",
        (0, 1, 2**32 - 1),
        (0, 1, 3),
    ),
    (
        "int64",
        (-(2**63), -1, 0, 1, 2**63 - 1),
        (6, 6, 0, 1, 0),
    ),
    (
        "uint64",
        (0, 1, 2**63, 2**64 - 1),
        (0, 1, 1, 1),
    ),
)


@pytest.mark.parametrize("backend_name", ("hybrid", "numba_cpu"))
def test_indexed_modulo_matches_sparse_assignment_reference_vectors(
    backend_name: str,
) -> None:
    provider = IndexedModuloPartition()

    for dtype, values, expected in _REFERENCE_VECTORS:
        keys = np.asarray(values, dtype=dtype)
        with use_backend(backend_name):
            partitions = provider(
                keys,
                dim_size=2**64,
                num_workers=7,
            )
        np.testing.assert_array_equal(np.asarray(partitions), expected)


def test_indexed_modulo_projects_logical_partitions_through_device_mesh_order() -> None:
    mesh = DeviceMesh(
        axis_names=("worker",),
        device_shape=(4,),
        device_ids=(7, 3, 11, 5),
    )
    provider = IndexedModuloPartition()
    logical_partitions = np.arange(4, dtype=np.int32)

    with use_backend("hybrid"):
        partition_ids = provider(
            logical_partitions,
            dim_size=2**64,
            num_workers=mesh.size,
        )
    physical_ranks = np.asarray(mesh.device_ids).reshape(-1)[
        np.asarray(partition_ids, dtype=np.intp)
    ]

    np.testing.assert_array_equal(partition_ids, [0, 1, 2, 3])
    np.testing.assert_array_equal(physical_ranks, [7, 3, 11, 5])
