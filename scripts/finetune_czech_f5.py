#!/usr/bin/env python3
"""Run F5-TTS finetune_cli with Vocos on the same device as the trainer.

F5's load_vocoder() defaults to MPS whenever Metal is available. CPU fine-tune
(ACCELERATE_USE_CPU=true) then dies at --log_samples:

    slow_conv2d_forward_mps: input(device='cpu') and weight(device='mps:0')

This wrapper pins Vocos to the training device and moves decode inputs onto
the vocoder's parameter device. argv is forwarded to finetune_cli.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import device as apple_device  # noqa: E402

apple_device.bootstrap_mps_fallback()


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def training_device() -> str:
    if _truthy_env("ACCELERATE_USE_CPU"):
        return "cpu"
    explicit = os.environ.get("F5_TRAIN_DEVICE", "").strip().lower()
    if explicit == "cpu":
        return "cpu"
    if explicit in {"mps", "auto"}:
        return apple_device.device_str(explicit)
    if explicit in {"cuda", "xpu"}:
        return explicit
    return apple_device.device_str("mps")


def align_module_input(module: Any, tensor: Any) -> Any:
    """Move `tensor` onto `module`'s first parameter device/dtype."""
    param = next(module.parameters())
    if tensor.device != param.device or tensor.dtype != param.dtype:
        return tensor.to(device=param.device, dtype=param.dtype)
    return tensor


def wrap_vocoder_device(vocoder: Any) -> Any:
    """Keep Vocos.decode / BigVGAN forward on the vocoder's own device."""
    if hasattr(vocoder, "decode"):
        orig_decode = vocoder.decode

        def decode(mel, *args, **kwargs):
            return orig_decode(align_module_input(vocoder, mel), *args, **kwargs)

        vocoder.decode = decode

    orig_forward = vocoder.forward

    def forward(mel, *args, **kwargs):
        return orig_forward(align_module_input(vocoder, mel), *args, **kwargs)

    vocoder.forward = forward
    return vocoder


def patch_load_vocoder() -> str:
    import f5_tts.infer.utils_infer as infer_utils

    orig = infer_utils.load_vocoder
    device = training_device()

    def load_vocoder(*args, **kwargs):
        kwargs["device"] = device
        return wrap_vocoder_device(orig(*args, **kwargs))

    infer_utils.load_vocoder = load_vocoder
    apple_device.log(f"F5-TTS train: Vocos device={device} (match training)")
    return device


def main() -> None:
    patch_load_vocoder()
    from f5_tts.train.finetune_cli import main as finetune_main

    finetune_main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
