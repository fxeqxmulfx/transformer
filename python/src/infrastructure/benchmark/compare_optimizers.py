"""Run tested GPU optimizer comparisons on reference GPTMini and Sparsemax.

From the repository root:
  .venv/bin/python -m gpt_mini.infrastructure.benchmark.compare_optimizers --calibrate
  .venv/bin/python -m gpt_mini.infrastructure.benchmark.compare_optimizers
"""

from gpt_mini.infrastructure.benchmark.paths import benchmark_path as _bench_path, DATA_FILE as _data_file

import argparse
import os

# Must be set before importing Torch or creating a CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["OPTIMIZER_BENCH_CUDA"] = "1"

from gpt_mini.infrastructure.benchmark.optimizer_benchmark.runner import launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=_data_file)
    parser.add_argument("--output", default=_bench_path('experiments/optimizer_benchmark/results/rtx3050'))
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--attention", nargs="+", choices=["softmax", "sparsemax"], default=["softmax", "sparsemax"])
    parser.add_argument("--screen-steps", type=int, default=250)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    launch(parser.parse_args())


if __name__ == "__main__":
    main()
