"""Portable fingerprints for the code used by modular training and analysis."""

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def source_hashes():
    directory = Path(__file__).resolve().parent
    paths = list(directory.glob("*.py")) + [ROOT / name for name in (
        "experiments/gpt_mini.py", "experiments/optimizer_benchmark/coordinate.py",
        "experiments/optimizer_benchmark/common.py", "experiments/synthetic_trainers/runtime.py",
        "experiments/synthetic_trainers/curves.py")]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}
