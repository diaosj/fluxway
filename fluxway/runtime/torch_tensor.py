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


def scalar_to_transport(value, *, dtype=None):
    import torch

    from fluxway.common.types.tensor import TransportTensor

    tensor = torch.tensor([value], dtype=dtype).cpu()
    return TransportTensor.from_tensor(tensor, global_shape=tuple(tensor.shape))

