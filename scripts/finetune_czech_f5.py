#!/usr/bin/env python3
"""Run F5-TTS finetune_cli with Mac-safe Vocos placement and DataLoader settings.

F5's load_vocoder() defaults to MPS whenever Metal is available. CPU fine-tune
(ACCELERATE_USE_CPU=true) then dies at --log_samples:

    slow_conv2d_forward_mps: input(device='cpu') and weight(device='mps:0')

F5 also builds the train DataLoader with num_workers=16 and persistent_workers.
Ansible-launched processes on macOS often inherit a 256-file soft limit, so
epoch 2's worker respawn fails with:

    OSError: [Errno 24] Too many open files

This wrapper pins Vocos to the training device, raises RLIMIT_NOFILE, and
forces an in-process DataLoader. argv is forwarded to finetune_cli.
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


def dataloader_workers() -> int:
    raw = os.environ.get("F5_TRAIN_NUM_WORKERS", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def raise_nofile_limit(minimum: int = 8192) -> int:
    """Raise the soft file-descriptor cap when Ansible inherited macOS' 256."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    inf = getattr(resource, "RLIM_INFINITY", -1)
    if hard in {inf, -1} or hard > 1_000_000:
        target = max(soft, minimum)
    else:
        target = max(soft, min(minimum, hard))
    if target > soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except (ValueError, OSError) as exc:
            apple_device.log(f"F5-TTS train: could not raise RLIMIT_NOFILE ({exc})")
    new_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    apple_device.log(f"F5-TTS train: RLIMIT_NOFILE soft={new_soft}")
    return new_soft


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


def patch_trainer_dataloader() -> int:
    """Force F5's train DataLoader off multiprocessing (avoids EMFILE on epoch 2)."""
    from torch.utils.data import DataLoader as TorchDataLoader

    import f5_tts.model.trainer as trainer_mod

    workers = dataloader_workers()

    class TrainDataLoader(TorchDataLoader):
        def __init__(self, *args, **kwargs):
            kwargs["num_workers"] = workers
            kwargs["persistent_workers"] = False
            if workers <= 0:
                kwargs["pin_memory"] = False
                kwargs.pop("prefetch_factor", None)
            super().__init__(*args, **kwargs)

    trainer_mod.DataLoader = TrainDataLoader

    orig_train = trainer_mod.Trainer.train

    def train(self, train_dataset, num_workers=16, resumable_with_seed=None):
        return orig_train(
            self,
            train_dataset,
            num_workers=workers,
            resumable_with_seed=resumable_with_seed,
        )

    trainer_mod.Trainer.train = train
    apple_device.log(
        f"F5-TTS train: DataLoader workers={workers} persistent_workers=False"
    )
    return workers


def main() -> None:
    raise_nofile_limit()
    patch_trainer_dataloader()
    patch_load_vocoder()
    from f5_tts.train.finetune_cli import main as finetune_main

    finetune_main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
