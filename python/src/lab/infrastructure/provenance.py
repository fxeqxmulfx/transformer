"""What determines a trajectory besides the experiment: the code, the framework, the device.

`engine` is what a continued run must share with the session that began it;
the rest is recorded for the reader.
"""

import hashlib
from pathlib import Path
import platform
import subprocess

import torch

PACKAGE = Path(__file__).resolve().parents[1]


def code():
    """sha256 of every lab source a run executes: all but the command line."""
    return {str(path.relative_to(PACKAGE)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(PACKAGE.rglob("*.py"))
            if path.relative_to(PACKAGE).parts[0] not in ("interfaces", "__main__.py")}


def device_name(device):
    """The GPU, or the CPU model the kernel reports, whose instructions compiled kernels use."""
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def commit():
    """The checked-out commit, and whether the lab sources differ from it."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE, capture_output=True, text=True,
                              check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=PACKAGE, capture_output=True,
                                text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"head": head, "lab_modified": bool(status.strip())}


def provenance(device):
    return {"engine": {"code": code(), "torch": torch.__version__, "device": device_name(device)},
            "commit": commit(), "cuda": torch.version.cuda, "python": platform.python_version()}
