from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import fluxway.common.tensor_backend as tensor_backend


class _FakeArray:
    def __getitem__(self, _index):
        return self

    def __gt__(self, _other):
        return self


class _ColdCompiler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._compiling = False
        self.ready = False

    def compile(self) -> None:
        with self._lock:
            if self.ready:
                return
            if self._compiling:
                raise RuntimeError("concurrent cold CuPy compilation")
            self._compiling = True
        try:
            time.sleep(0.02)
        finally:
            with self._lock:
                self.ready = True
                self._compiling = False


class _FakeCupy:
    int32 = np.int32

    def __init__(self) -> None:
        self.compiler = _ColdCompiler()
        self.synchronize_calls = 0
        self.cuda = SimpleNamespace(
            Stream=SimpleNamespace(null=SimpleNamespace(synchronize=self._synchronize))
        )

    def _synchronize(self) -> None:
        self.synchronize_calls += 1

    def arange(self, _size: int, *, dtype):
        assert dtype is self.int32
        self.compiler.compile()
        return _FakeArray()

    @staticmethod
    def min(_value) -> bool:
        return True


def _cuda_backend(fake_cupy: _FakeCupy):
    backend = tensor_backend.HybridBackend(default_device="cpu", enable_numba_jit=False)
    backend._default_device = "cuda"
    backend._cp = fake_cupy
    return backend


def _comparison(fake_cupy: _FakeCupy) -> bool:
    values = fake_cupy.arange(2, dtype=fake_cupy.int32)
    return fake_cupy.min(values[1:] > values[:-1])


def _concurrent_results(fake_cupy: _FakeCupy) -> list[object]:
    barrier = threading.Barrier(5)

    def run() -> object:
        barrier.wait(timeout=1)
        try:
            return _comparison(fake_cupy)
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=5) as executor:
        return list(executor.map(lambda _unused: run(), range(5)))


def test_worker_runtime_initialization_serializes_cupy_first_compile() -> None:
    cold_cupy = _FakeCupy()
    cold_results = _concurrent_results(cold_cupy)
    assert any(isinstance(result, RuntimeError) for result in cold_results)

    warmed_cupy = _FakeCupy()
    backend = _cuda_backend(warmed_cupy)
    backend.initialize_worker_runtime()
    warm_results = _concurrent_results(warmed_cupy)

    assert warm_results == [True] * 5
    assert warmed_cupy.synchronize_calls == 1


def test_worker_runtime_initialization_is_idempotent_for_cuda() -> None:
    fake_cupy = _FakeCupy()
    backend = _cuda_backend(fake_cupy)

    backend.initialize_worker_runtime()
    backend.initialize_worker_runtime()

    assert fake_cupy.synchronize_calls == 1


def test_worker_runtime_initialization_compiles_all_fluxway_numba_helpers(
    monkeypatch,
) -> None:
    backend = tensor_backend.HybridBackend(default_device="cpu", enable_numba_jit=False)
    backend._use_jit = True
    calls: list[str] = []

    monkeypatch.setattr(
        tensor_backend,
        "_numba_bincount_loop",
        lambda x, out: calls.append("bincount"),
    )
    monkeypatch.setattr(
        tensor_backend,
        "_numba_bincount_weighted_loop",
        lambda x, weights, out: calls.append("weighted_bincount"),
    )
    monkeypatch.setattr(
        tensor_backend,
        "_numba_cumsum_loop",
        lambda x, out: calls.append("cumsum"),
    )
    monkeypatch.setattr(
        tensor_backend,
        "_numba_take_loop",
        lambda data, indices, out: calls.append("take"),
    )

    backend.initialize_worker_runtime()

    assert calls == ["bincount", "weighted_bincount", "cumsum", "take"]


def test_set_hybrid_device_initializes_before_publishing_backend(monkeypatch) -> None:
    published_before_initialization: list[bool] = []
    previous_backend = tensor_backend._BACKEND

    class _Backend:
        def __init__(self, **_kwargs) -> None:
            pass

        def initialize_worker_runtime(self) -> None:
            published_before_initialization.append(tensor_backend._BACKEND is self)

    monkeypatch.setattr(tensor_backend, "HybridBackend", _Backend)
    try:
        tensor_backend.set_hybrid_device("cuda")
        assert published_before_initialization == [False]
    finally:
        tensor_backend._BACKEND = previous_backend


def test_set_hybrid_device_propagates_runtime_initialization_failure(
    monkeypatch,
) -> None:
    previous_backend = tensor_backend._BACKEND

    class _Backend:
        def __init__(self, **_kwargs) -> None:
            pass

        def initialize_worker_runtime(self) -> None:
            raise RuntimeError("compiler setup failed")

    monkeypatch.setattr(tensor_backend, "HybridBackend", _Backend)
    try:
        with pytest.raises(RuntimeError, match="compiler setup failed"):
            tensor_backend.set_hybrid_device("cuda")
        assert tensor_backend._BACKEND is previous_backend
    finally:
        tensor_backend._BACKEND = previous_backend
