"""CUDA helpers for Fluxway examples."""


def get_worker_cuda_device():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TE minimal example.")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    return device


def get_cuda_debug_info():
    import os
    import torch

    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "current_device": torch.cuda.current_device()
        if torch.cuda.is_available()
        else None,
        "device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
    }

