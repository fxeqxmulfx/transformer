"""Fingerprint the compiler adapter and its preserved measured dependencies."""

import hashlib
from pathlib import Path

from experiments.compiled_benchmark.protocol import source_hashes as previous_sources


def source_hashes():
    result = previous_sources()
    for path in sorted(Path("experiments/full_compile_benchmark").rglob("*.py")):
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def preflight_hashes():
    directory = Path("experiments/full_compile_benchmark/results/preflight")
    names = ("tests.log", "training_tests.log", "throughput.json", "stdout.log")
    return {str(directory / name): hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in names}
