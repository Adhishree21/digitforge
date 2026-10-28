"""Device selection: CUDA when a CUDA-enabled PyTorch build is installed, otherwise CPU."""
import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_info() -> dict:
    dev = get_device()
    info = {"device": str(dev), "torch": torch.__version__, "cuda_build": torch.version.cuda}
    if dev.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        info["name"] = props.name
        info["vram_gb"] = round(props.total_memory / 1024**3, 1)
    else:
        info["name"] = "CPU (no CUDA-enabled PyTorch detected)"
    return info


def sync(device: torch.device) -> None:
    """Block until queued GPU work finishes (needed for honest timing)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
