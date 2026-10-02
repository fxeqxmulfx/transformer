"""Frozen, single-mechanism follow-up to Convexifying Transformers, Section 4."""

from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch

from .paper_reproduction.provenance import ROOT, source_hashes


def calibration_recipes(base):
    if base.model != "gptmini" or base.optimizer != "amsgradw":
        raise ValueError("Stability calibration requires the raw AMSGradW GPTMini control")
    controls = [("short-lr001", .001, "short_final"),
                ("wrap-lr001", .001, "wrap_epoch"),
                ("short-lr0003", .0003, "short_final"),
                ("short-lr0001", .0001, "short_final")]
    return [{"name": name, "config": asdict(replace(base, learning_rate=rate, batch_policy=policy)),
             "changed_mechanism": "unchanged_model_control" if index == 0 else
                "batch_policy" if policy == "wrap_epoch" else "learning_rate"}
            for index, (name, rate, policy) in enumerate(controls)]


def frozen_sources():
    hashes = source_hashes()
    for name in ("stability.py", "stability_protocol.py", "persistence.py", "paper_phases.py"):
        path = Path(__file__).parent / name
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(hashes.items()))


def environment(device):
    cuda = device.startswith("cuda")
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda, "device": device,
            "gpu": torch.cuda.get_device_name(device) if cuda else None,
            "compute_capability": list(torch.cuda.get_device_capability(device)) if cuda else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory if cuda else None,
            "cuda_arches": torch.cuda.get_arch_list() if cuda else [],
            "cudnn": torch.backends.cudnn.version(),
            "platform": platform.platform(), "threads": 1,
            "driver": subprocess.check_output([
                "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip() if cuda else None}


def paper_fingerprints():
    paths = [ROOT / "papers/arXiv-2211.11052v1/arxiv.tex",
             ROOT / "papers/arXiv-1912.02292v1/rffs.tex"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}
