import numpy as np
import pytest

from fluxway.common.allocation import (
    AllocationRegistry,
    AllocationRelation,
    AllocationIdentity,
)
from fluxway.common.coherence import (
    CoherenceSubject,
    CoherenceInstance,
    UnsupportedAllocationHandoffError,
)
from fluxway.common.tensor_backend import set_backend
from fluxway.common.tensor_backend import HybridBackend
from fluxway.common.types.tensor import (
    runtime_allocation_descriptor,
    runtime_allocation_relation,
)


@pytest.fixture(autouse=True)
def use_numpy_backend():
    set_backend("numba_cpu")


def test_numpy_allocation_relation_distinguishes_exact_overlap_and_copy() -> None:
    values = np.arange(8, dtype=np.int64)

    assert (
        runtime_allocation_relation(values, values.reshape(2, 4))
        is AllocationRelation.EXACT
    )
    assert runtime_allocation_relation(values, values[1:]) is AllocationRelation.OVERLAP
    assert (
        runtime_allocation_relation(values, values.copy())
        is AllocationRelation.DISJOINT
    )


def test_handoff_atomically_moves_exact_binding() -> None:
    values = np.arange(8, dtype=np.int64)
    old = CoherenceInstance(CoherenceSubject(1, 0, 7), 3)
    new = CoherenceInstance(CoherenceSubject(1, 0, 8), 3)
    registry = AllocationRegistry()
    registry.bind(old, runtime_allocation_descriptor(values))

    binding = registry.handoff(
        old,
        new,
        runtime_allocation_descriptor(values.reshape(2, 4)),
    )

    assert binding.instance == new
    assert binding.descriptor.owner is values
    with pytest.raises(UnsupportedAllocationHandoffError):
        registry.binding(old)


def test_handoff_rejects_partial_overlap() -> None:
    values = np.arange(8, dtype=np.int64)
    old = CoherenceInstance(CoherenceSubject(1, 0, 7), 3)
    new = CoherenceInstance(CoherenceSubject(1, 0, 8), 3)
    registry = AllocationRegistry()
    registry.bind(old, runtime_allocation_descriptor(values))

    with pytest.raises(UnsupportedAllocationHandoffError, match="exact"):
        registry.handoff(old, new, runtime_allocation_descriptor(values[1:]))


def test_handoff_tombstone_requires_explicit_forget() -> None:
    values = np.arange(8, dtype=np.int64)
    instance = CoherenceInstance(CoherenceSubject(1, 0, 7), 3)
    new = CoherenceInstance(CoherenceSubject(1, 0, 8), 3)
    registry = AllocationRegistry()
    registry.bind(instance, runtime_allocation_descriptor(values))
    registry.handoff(instance, new, runtime_allocation_descriptor(values))
    registry.forget_retired(instance)
    assert registry.is_bound(new)


def test_forget_retired_removes_handoff_tombstone_without_watermark() -> None:
    values = np.arange(8, dtype=np.int64)
    old = CoherenceInstance(CoherenceSubject(1, 0, 7), 3)
    new = CoherenceInstance(CoherenceSubject(1, 0, 8), 3)
    registry = AllocationRegistry()
    registry.bind(old, runtime_allocation_descriptor(values))
    registry.handoff(old, new, runtime_allocation_descriptor(values))
    registry.forget_retired(old)
    assert registry.is_bound(new)


def test_hybrid_backend_torch_allocation_descriptor() -> None:
    """HybridBackend preserves Torch storage identity and tensor byte span."""

    class FakeStorage:
        def data_ptr(self) -> int:
            return 10_000

        def nbytes(self) -> int:
            return 128

    class FakeTensor:
        is_cuda = False
        device = "cpu"

        def is_contiguous(self) -> bool:
            return True

        def untyped_storage(self) -> FakeStorage:
            return FakeStorage()

        def data_ptr(self) -> int:
            return 10_032

        def nelement(self) -> int:
            return 8

        def element_size(self) -> int:
            return 4

    class FakeTorch:
        Tensor = FakeTensor

    backend = HybridBackend(default_device="cpu", enable_numba_jit=False)
    backend._torch = FakeTorch
    descriptor = backend.allocation_descriptor(FakeTensor())

    assert descriptor.owner.data_ptr() == 10_000
    assert descriptor.identity == AllocationIdentity(
        "torch", "cpu", 10_000, 128, 10_032, 32
    )


def test_hybrid_backend_rejects_noncontiguous_torch_tensor() -> None:
    class FakeTensor:
        is_cuda = False

        def is_contiguous(self) -> bool:
            return False

    class FakeTorch:
        Tensor = FakeTensor

    backend = HybridBackend(default_device="cpu", enable_numba_jit=False)
    backend._torch = FakeTorch

    with pytest.raises(UnsupportedAllocationHandoffError, match="contiguous"):
        backend.allocation_descriptor(FakeTensor())
