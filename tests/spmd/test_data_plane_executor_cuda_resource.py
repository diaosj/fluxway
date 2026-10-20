from fluxway.spmd.worker.data_plane_executor import _object_requires_cuda_visible


class FakeCudaArray:
    __cuda_array_interface__ = {"shape": (1,), "typestr": "<f4", "data": (1, False)}


class FakeDlpackCudaArray:
    def __dlpack_device__(self):
        return (2, 0)


class FakeTransportTensor:
    def __init__(self, flat_values):
        self.flat_values = flat_values
        self.nested_splits = []
        self.routing_indices = None
        self.origin_indices = None
        self.receipt = None


def test_detects_cuda_array_interface_in_nested_payload():
    payload = {"result": [FakeTransportTensor(FakeCudaArray())]}

    assert _object_requires_cuda_visible(payload)


def test_detects_cuda_dlpack_device_in_nested_payload():
    payload = (FakeTransportTensor(FakeDlpackCudaArray()),)

    assert _object_requires_cuda_visible(payload)


def test_cpu_payload_does_not_require_cuda_visible():
    payload = {"result": [FakeTransportTensor([1, 2, 3])]}

    assert not _object_requires_cuda_visible(payload)
