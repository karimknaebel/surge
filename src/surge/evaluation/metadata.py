import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


def collect_metadata(device: torch.device) -> dict:
    gpu = None
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        gpu = torch.cuda.get_device_name(device)

    commit = dirty = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "diff", "HEAD"],
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL,
            )
        )
    except (OSError, subprocess.SubprocessError):
        pass

    return {
        "started_at": datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "hostname": socket.gethostname(),
        "device": str(device),
        "gpu": gpu,
        "command": list(sys.orig_argv),
        "git": {"commit": commit, "dirty": dirty},
    }
