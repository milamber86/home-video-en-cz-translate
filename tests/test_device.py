"""Tests for device helpers (no torch required)."""

from __future__ import annotations

import os

import device


def test_bootstrap_sets_mps_fallback_env():
    os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    device.bootstrap_mps_fallback()
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_log_writes_to_stderr(capsys):
    device.log("hello pipeline")
    captured = capsys.readouterr()
    assert captured.err.strip() == "hello pipeline"


def test_module_import_enables_mps_fallback():
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
