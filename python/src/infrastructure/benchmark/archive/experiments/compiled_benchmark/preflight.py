"""Run analytic and CUDA compiler tests before any compiled experiment."""

import os
from pathlib import Path
import unittest

root = Path(__file__).resolve().parents[2]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["OPTIMIZER_BENCH_CUDA"] = "1"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "experiments/runs/inductor_cache"))
os.environ.setdefault("TRITON_CACHE_DIR", str(root / "experiments/runs/triton_cache"))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")

import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output = root / "experiments/compiled_benchmark/results/preflight"
    output.mkdir(parents=True, exist_ok=True)
    suite = unittest.TestLoader().discover("experiments/compiled_benchmark/tests", top_level_dir=".")
    with (output / "tests.log").open("w") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(f"Tests={result.testsRun} failures={len(result.failures)} errors={len(result.errors)} skipped={len(result.skipped)}", flush=True)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError(f"Compiler preflight failed; see {output / 'tests.log'}")


if __name__ == "__main__":
    main()
