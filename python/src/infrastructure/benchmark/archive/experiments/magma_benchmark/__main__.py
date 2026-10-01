"""GPU-only Magma comparison, reusing the frozen GPTMini protocol."""

import argparse
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["OPTIMIZER_BENCH_CUDA"] = "1"

from .runner import launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/magma_benchmark/results/rtx3050")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    launch(parser.parse_args())


if __name__ == "__main__":
    main()
