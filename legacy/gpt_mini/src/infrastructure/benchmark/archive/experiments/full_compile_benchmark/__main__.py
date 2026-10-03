"""Compare optimizers on CUDA until validation plateau/divergence."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["OPTIMIZER_BENCH_CUDA"] = "1"

root = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "experiments/runs/inductor_cache"))
os.environ.setdefault("TRITON_CACHE_DIR", str(root / "experiments/runs/triton_cache"))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")

from .runner import launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--every", type=int, default=250)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument("--min-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--divergence-delta", type=float, default=0.1)
    parser.add_argument("--divergence-patience", type=int, default=3)
    parser.add_argument("--tests-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    launch(parser.parse_args())


if __name__ == "__main__":
    main()
