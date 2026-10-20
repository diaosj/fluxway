import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from fluxway.common.tensor_backend import prepare_dlpack_input


class _ReadonlyFlags:
    @property
    def writeable(self):
        return False

    @writeable.setter
    def writeable(self, value):
        raise ValueError("readonly view flag")


class _CudaArray:
    flags = _ReadonlyFlags()
    __cuda_array_interface__ = {
        "shape": (1,),
        "typestr": "<f4",
        "data": (16, True),
        "version": 3,
    }


@pytest.fixture(params=[False, True], ids=["foreign", "cupy"])
def cupy_adoption(monkeypatch, request):
    module = ModuleType("cupy")
    module.ndarray = _CudaArray if request.param else type("CupyArray", (), {})
    monkeypatch.setitem(sys.modules, "cupy", module)
    return module, _CudaArray()


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
def test_cuda_adoption_errors_propagate(cupy_adoption, error_type):
    module, source = cupy_adoption
    error = error_type("CUDA adoption failed")
    attempts = []

    def fail(view):
        attempts.append(view)
        raise error

    module.asarray = fail

    with pytest.raises(error_type, match="CUDA adoption failed") as caught:
        prepare_dlpack_input(source)

    assert caught.value is error
    assert len(attempts) == 1


def test_cuda_adoption_requires_its_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)

    with pytest.raises(ModuleNotFoundError, match="cupy"):
        prepare_dlpack_input(_CudaArray())


@pytest.mark.parametrize("numpy_input", [False, True])
def test_cpu_and_other_inputs_do_not_require_cupy(monkeypatch, numpy_input):
    monkeypatch.setitem(sys.modules, "cupy", None)
    source = (
        np.arange(4)
        if numpy_input
        else SimpleNamespace(flags=SimpleNamespace(writeable=False))
    )
    if numpy_input:
        source.flags.writeable = False

    result = prepare_dlpack_input(source)

    if numpy_input:
        assert result.flags.writeable
        assert np.shares_memory(source, result)
    else:
        assert result is source


def test_cuda_adoption_retains_source_and_pointer(cupy_adoption):
    module, source = cupy_adoption
    adopted = object()
    views = []

    def asarray(view):
        views.append(view)
        return adopted

    module.asarray = asarray

    assert prepare_dlpack_input(source) is adopted
    assert len(views) == 1
    assert views[0]._source is source
    expected = dict(source.__cuda_array_interface__, data=(16, False))
    assert views[0].__cuda_array_interface__ == expected
    assert source.__cuda_array_interface__["data"] == (16, True)
