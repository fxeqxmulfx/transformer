"""Fingerprint the compiler adapter and its preserved measured dependencies."""

from gpt_mini.infrastructure.benchmark.paths import benchmark_path as _bench_path
from gpt_mini.infrastructure.benchmark.provenance import source_hashes as _source_hashes

import hashlib
from pathlib import Path

from gpt_mini.infrastructure.benchmark.compiled_benchmark.protocol import source_hashes as previous_sources


def source_hashes():
    return _source_hashes()


def preflight_hashes():
    directory = Path(_bench_path('experiments/full_compile_benchmark/results/preflight'))
    names = ("tests.log", "training_tests.log", "throughput.json", "stdout.log")
    return {str(directory / name): hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in names}
