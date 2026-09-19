#!/usr/bin/env python3
"""Debug probe: Ansible localhost interpreter / Rosetta / xcrun arch mismatch."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

LOG = Path("/Users/otto/Cursor_Repos/home-video-en-cz-translate/.cursor/debug-e0bbd4.log")
SESSION = "e0bbd4"


def emit(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # region agent log
    payload = {
        "sessionId": SESSION,
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")
    # endregion


def run(cmd: list[str]) -> dict:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return {
            "cmd": cmd,
            "rc": proc.returncode,
            "stdout": (proc.stdout or "")[:800],
            "stderr": (proc.stderr or "")[:800],
        }
    except Exception as exc:
        return {"cmd": cmd, "rc": -1, "stdout": "", "stderr": str(exc)}


def main() -> int:
    emit(
        "B",
        "debug_ansible_arch.py:probe_self",
        "Current interpreter identity",
        {
            "executable": sys.executable,
            "platform_machine": platform.machine(),
            "uname_machine": os.uname().machine,
            "version": sys.version,
        },
    )

    brew = shutil.which("brew")
    emit(
        "C",
        "debug_ansible_arch.py:probe_brew",
        "Homebrew prefix and brew path",
        {
            "which_brew": brew,
            "opt_homebrew_exists": Path("/opt/homebrew/bin/brew").exists(),
            "usr_local_brew_exists": Path("/usr/local/bin/brew").exists(),
            "brew_prefix": run([brew, "--prefix"]) if brew else None,
        },
    )

    emit(
        "E",
        "debug_ansible_arch.py:probe_clt_python",
        "Command Line Tools python file type",
        {
            "usr_bin_python3": run(["file", "/usr/bin/python3"]),
            "clt_python": run(
                ["file", "/Library/Developer/CommandLineTools/usr/bin/python3"]
            ),
        },
    )

    shim = run(
        ["/usr/bin/python3", "-c", "import platform; print(platform.machine())"]
    )
    emit(
        "A",
        "debug_ansible_arch.py:probe_usr_bin_python3",
        "Spawn /usr/bin/python3 from this process (Ansible inventory interpreter)",
        shim,
    )

    arm = run(
        [
            "arch",
            "-arm64",
            "/usr/bin/python3",
            "-c",
            "import platform,sys; print(platform.machine(), sys.executable)",
        ]
    )
    emit(
        "D",
        "debug_ansible_arch.py:probe_arch_arm64",
        "Forced arm64 /usr/bin/python3",
        arm,
    )

    xcrun_in_err = "libxcrun" in (shim.get("stderr") or "")
    print(
        json.dumps(
            {
                "self_machine": platform.machine(),
                "usr_bin_python3_rc": shim.get("rc"),
                "xcrun_error": xcrun_in_err,
                "arm64_python_rc": arm.get("rc"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
