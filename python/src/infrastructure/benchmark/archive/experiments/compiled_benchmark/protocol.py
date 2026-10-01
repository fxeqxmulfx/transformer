"""Fingerprint the compiler adapter and its preserved measured dependencies."""

import hashlib
from pathlib import Path

from experiments.patience_benchmark.protocol import source_hashes as previous_sources


def source_hashes():
    result = previous_sources()
    for path in sorted(Path("experiments/compiled_benchmark").rglob("*.py")):
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def preflight_hashes():
    directory = Path("experiments/compiled_benchmark/results/preflight")
    names = ("tests.log", "throughput.json", "environment.json", "recompilation.json", "recompiles.log")
    return {str(directory / name): hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in names}
