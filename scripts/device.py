"""Apple Silicon device helpers.

Import this module before any `import torch` so MPS CPU fallback is enabled.
Whisper in this project uses MLX (Metal), not torch.device("mps").
"""

from __future__ import annotations

import os
import sys
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def bootstrap_mps_fallback() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available() and torch.backends.mps.is_built())


def resolve_torch_device(prefer: str = "mps") -> Any:
    """Return a torch.device. `prefer` is mps|cpu."""
    import torch

    prefer = (prefer or "mps").strip().lower()
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in {"mps", "auto"} and mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_str(prefer: str = "mps") -> str:
    return str(resolve_torch_device(prefer))


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
