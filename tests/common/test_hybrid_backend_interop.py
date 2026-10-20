import numpy as np
import pytest

from fluxway.common.tensor_backend import HybridBackend


def test_unavailable_cuda_request_never_returns_cpu_array():
    backend = HybridBackend(default_device="cpu", enable_numba_jit=False)
    backend._cp = None
    with pytest.raises(RuntimeError, match="CUDA|CuPy"):
        backend.asarray(np.arange(3), device="cuda", copy=False)


def test_cpu_jax_export_preserves_allocation_and_values():
    import jax

    backend = HybridBackend(default_device="cpu", enable_numba_jit=False)
    source = jax.device_put(np.arange(8, dtype=np.float32), jax.devices("cpu")[0])
    result = backend.asarray(source, device="cpu", copy=False)
    np.testing.assert_array_equal(result, np.arange(8, dtype=np.float32))
    assert result.__array_interface__["data"][0] == source.unsafe_buffer_pointer()


def test_foreign_device_mismatch_fails_before_export():
    class CudaExport:
        def __dlpack_device__(self):
            return 2, 0

        def __dlpack__(self, **kwargs):
            raise AssertionError("mismatched device reached the exporter")

    backend = HybridBackend(default_device="cpu", enable_numba_jit=False)
    with pytest.raises(ValueError, match="device|CUDA|CPU"):
        backend.asarray(CudaExport(), device="cpu", copy=False)


def test_export_error_is_propagated_without_conversion_fallback():
    class BrokenExport:
        def __dlpack_device__(self):
            return 1, 0

        def __dlpack__(self, **kwargs):
            raise RuntimeError("broken exporter")

        def __array__(self, **kwargs):
            raise AssertionError("fallback conversion was attempted")

    backend = HybridBackend(default_device="cpu", enable_numba_jit=False)
    with pytest.raises(RuntimeError, match="broken exporter"):
        backend.asarray(BrokenExport(), device="cpu", copy=False)


def test_dense_export_uses_cpu_jax_platform_and_preserves_values():
    import jax
    from fluxway.example.recommendation.dense_spmd import _local_jax_to_native

    source = jax.device_put(np.arange(8, dtype=np.float32), jax.devices("cpu")[0])
    result = _local_jax_to_native(source)
    assert isinstance(result, np.ndarray)
    assert result.__array_interface__["data"][0] == source.unsafe_buffer_pointer()
    np.testing.assert_array_equal(result, np.arange(8, dtype=np.float32))


def test_dense_export_rejects_non_jax_payload():
    from fluxway.example.recommendation.dense_spmd import _local_jax_to_native

    with pytest.raises(TypeError, match="JAX"):
        _local_jax_to_native(np.arange(3))
