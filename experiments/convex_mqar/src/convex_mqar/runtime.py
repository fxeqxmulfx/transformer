"""Preserve execution provenance when resuming with a different kernel backend."""

import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch


FINGERPRINTED_FILES = ("rope.py", "engine.py", "data.py", "convex.py", "config.py",
                       "certify.py", "kernels.py", "runtime.py", "benchmark.py", "cli.py")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def record_execution(output, config, compiled, extra_files=()):
    fingerprinted_files = (*FINGERPRINTED_FILES, *extra_files)
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "cuda": torch.version.cuda, "numpy": np.__version__,
                   "device": torch.cuda.get_device_name() if config.device == "cuda" else "cpu",
                   "precision": config.precision, "rope_base": config.rope_base,
                   "compiled_loss": compiled, "compile_backend": "inductor" if compiled else None,
                   "fingerprinted_files": list(fingerprinted_files)}
    environment["source_sha256"] = hashlib.sha256(b"".join(
        Path(__file__).with_name(name).read_bytes() for name in fingerprinted_files)).hexdigest()
    history_path = output / "execution_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    environment_path = output / "environment.json"
    if environment_path.exists() and not history:
        history.append({"time_unix": environment_path.stat().st_mtime,
                        "environment": json.loads(environment_path.read_text()),
                        "scope": "Original execution before provenance history was introduced"})
    status_path = output / "status.json"
    history.append({"time_unix": time.time(), "pid": os.getpid(), "environment": environment,
                    "previous_status": json.loads(status_path.read_text()) if status_path.exists() else None})
    write_json(history_path, history)
    write_json(environment_path, environment)
    return environment, history
