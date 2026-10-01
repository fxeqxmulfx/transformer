"""Run analytic and CUDA compiler tests before any compiled experiment."""

from gpt_mini.infrastructure.benchmark.paths import benchmark_path as _bench_path, WORK_ROOT as _work_root
from gpt_mini.infrastructure.benchmark.suites import test_suite as _test_suite

import os
from pathlib import Path
import unittest

root = _work_root
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["OPTIMIZER_BENCH_CUDA"] = "1"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / _bench_path('experiments/runs/inductor_cache')))
os.environ.setdefault("TRITON_CACHE_DIR", str(root / _bench_path('experiments/runs/triton_cache')))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")

import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output = root / _bench_path('experiments/compiled_benchmark/results/preflight')
    output.mkdir(parents=True, exist_ok=True)
    suite = _test_suite(('compiled_benchmark',))
    with (output / "tests.log").open("w") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(f"Tests={result.testsRun} failures={len(result.failures)} errors={len(result.errors)} skipped={len(result.skipped)}", flush=True)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError(f"Compiler preflight failed; see {output / 'tests.log'}")


if __name__ == "__main__":
    main()
