"""Command line interface for the standalone uv project."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="Convex recall versus a trained RoPE Transformer")
    commands = parser.add_subparsers(dest="command", required=True)
    checks = commands.add_parser("check", help="Verify model semantics before training")
    checks.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    benchmark = commands.add_parser("benchmark", help="Train and compare on one common MQAR test")
    benchmark.add_argument("--profile", choices=("full", "capacity", "smoke", "sanity"), default="full")
    benchmark.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    benchmark.add_argument("--output", type=Path)
    benchmark.add_argument("--data-root", type=Path, default=Path("data"))
    benchmark.add_argument("--compile", action="store_true", help="Compile training loss with TorchInductor")
    commands.add_parser("certify", help="Reproduce the construction's numerical recall checks")
    args, extra = parser.parse_known_args()
    if args.command == "certify":
        from .certify import main as certify
        sys.argv = [sys.argv[0], *extra]
        certify()
        return
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if args.command == "check":
        from .checks import check
        check(args.device)
        return
    from .benchmark import run
    from .config import profile
    config = replace(profile(args.profile), device=args.device,
                     precision="bf16" if args.device == "cuda" else "fp32")
    run(config, args.output or Path("runs") / args.profile, args.data_root, compiled=args.compile)
