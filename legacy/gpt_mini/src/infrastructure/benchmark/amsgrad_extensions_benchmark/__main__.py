"""Run the independent AMSGradW/MD comparison on CUDA."""

import argparse
import os

os.environ["OPTIMIZER_BENCH_CUDA"] = "1"

from .runner import launch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output")
parser.add_argument("--tests-only",action="store_true")
parser.add_argument("--screen-only",action="store_true")
parser.add_argument("--validate-only",action="store_true")
launch(parser.parse_args())
