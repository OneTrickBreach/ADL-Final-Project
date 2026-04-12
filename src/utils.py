"""Shared utilities — device detection and configuration."""

import torch


def get_device() -> torch.device:
    """Auto-detect the best available compute device.

    Priority: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU.

    Returns:
        torch.device ready to use with .to(device) and tensor creation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info(device: torch.device) -> str:
    """Return a human-readable description of the device."""
    if device.type == "cuda":
        return f"CUDA — {torch.cuda.get_device_name(device)}"
    if device.type == "mps":
        return "MPS — Apple Silicon GPU (Metal Performance Shaders)"
    return "CPU"
