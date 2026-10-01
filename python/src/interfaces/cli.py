"""Compose the benchmark application with CUDA and filesystem adapters."""

import argparse
import json
import os

from ..application.benchmark import RunBenchmark
from ..domain.benchmark import ModelConfig, Request
from ..domain.stopping import StopConfig


def parser():
    result = argparse.ArgumentParser(description="Benchmark TinyShakespeare with Softmax/Sparsemax; AMSGradW is the default optimizer.")
    result.add_argument("--output")
    optimizers = result.add_mutually_exclusive_group()
    optimizers.add_argument("--only", nargs="+", default=Request().methods,
                           help="Optimizers to run (default: amsgradw).")
    optimizers.add_argument("--all", dest="only", action="store_const", const=(),
                           default=argparse.SUPPRESS, help="Compare all 27 optimizers.")
    result.add_argument("--attention", nargs="+", choices=["softmax", "sparsemax"], default=["softmax", "sparsemax"])
    result.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    result.add_argument("--batch", type=int, default=32)
    result.add_argument("--width", type=int, default=128)
    result.add_argument("--layers", type=int, default=2)
    result.add_argument("--heads", type=int, default=4)
    result.add_argument("--context", type=int, default=64)
    result.add_argument("--every", type=int, default=250)
    result.add_argument("--patience", type=int, default=8)
    result.add_argument("--min-delta", type=float, default=.0001)
    result.add_argument("--min-steps", type=int, default=1000)
    result.add_argument("--max-steps", type=int, default=20000)
    result.add_argument("--divergence-delta", type=float, default=.1)
    result.add_argument("--divergence-patience", type=int, default=3)
    modes = result.add_mutually_exclusive_group()
    modes.add_argument("--tests-only", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    result.add_argument("--allow-partial", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["OPTIMIZER_BENCH_CUDA"] = "1"
    from ..infrastructure.results import FilesystemResults
    from ..infrastructure.training import CudaTraining

    model = ModelConfig(n_layers=args.layers, n_heads=args.heads, d_model=args.width,
                        d_ff=4 * args.width, max_seq_len=args.context)
    stopping = StopConfig(args.every, args.patience, args.min_delta, args.min_steps,
                          args.max_steps, args.divergence_delta, args.divergence_patience)
    request = Request(tuple(args.only), tuple(args.attention), tuple(args.seeds), args.batch,
                      model, stopping, args.tests_only, args.validate_only, args.allow_partial)
    results = FilesystemResults(args.output)
    outcome = RunBenchmark(CudaTraining(results.progress), results).execute(request)
    print(json.dumps(outcome, indent=2), flush=True)
