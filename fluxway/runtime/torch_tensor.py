"""Small PyTorch tensor bridge for Fluxway TransportTensor values."""


def transport_to_torch(tensor, *, device=None, dtype=None):
    import numpy as np
    import torch

    from fluxway.common.types.tensor import TransportTensor

    value = tensor.flat_values if isinstance(tensor, TransportTensor) else tensor

    if isinstance(value, torch.Tensor):
        out = value
    else:
        array = np.asarray(value)
        if not array.flags.writeable:
            array = array.copy()
        out = torch.as_tensor(array)

    to_kwargs = {}
    if device is not None:
        to_kwargs["device"] = device
    if dtype is not None:
        to_kwargs["dtype"] = dtype
    if to_kwargs:
        out = out.to(**to_kwargs)

    return out


def scalar_to_transport(value, *, dtype=None, global_shape=None):
    import numpy as np

    from fluxway.common.types.tensor import TransportTensor

    np_dtype = None
    if dtype is not None:
        dtype_name = str(dtype)
        if dtype_name.endswith("float32"):
            np_dtype = np.float32
        elif dtype_name.endswith("float64"):
            np_dtype = np.float64
        elif dtype_name.endswith("int64"):
            np_dtype = np.int64
        elif dtype_name.endswith("int32"):
            np_dtype = np.int32

    tensor = np.asarray([value], dtype=np_dtype)
    return TransportTensor.from_tensor(
        tensor, global_shape=global_shape or tuple(tensor.shape)
    )

