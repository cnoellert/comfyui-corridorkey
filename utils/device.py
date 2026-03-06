import torch


def get_device():
    """Resolve best available device, preferring ComfyUI's management."""
    try:
        import comfy.model_management
        return comfy.model_management.get_torch_device()
    except ImportError:
        pass

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device):
    """Select appropriate dtype for device. Avoids bfloat16 on MPS."""
    if device.type == "mps":
        return torch.float16
    return torch.float16


def clear_cache(device):
    """Free device memory cache."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
