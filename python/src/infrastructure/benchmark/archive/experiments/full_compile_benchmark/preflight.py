"""Run all-method full-graph contracts before any full-step benchmark."""

import os
from pathlib import Path
import unittest

root = Path(__file__).resolve().parents[2]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["OPTIMIZER_BENCH_CUDA"] = "1"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "experiments/runs/inductor_cache"))
os.environ.setdefault("TRITON_CACHE_DIR", str(root / "experiments/runs/triton_cache"))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")


def main():
    path = root / "experiments/full_compile_benchmark/results/preflight/tests.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = unittest.TestLoader().discover("experiments/full_compile_benchmark/tests", top_level_dir=".")
    with path.open("w") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(f"Tests={result.testsRun} failures={len(result.failures)} errors={len(result.errors)} skipped={len(result.skipped)}", flush=True)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError(f"Full-step preflight failed: {path}")


if __name__ == "__main__":
    main()
