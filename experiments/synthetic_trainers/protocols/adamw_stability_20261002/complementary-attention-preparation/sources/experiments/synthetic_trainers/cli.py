"""Run or validate the synthetic RASP-family suite from the repository root."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .config import ModelSpec, TrainConfig
from .data import build_split
from .specs import CONTROL_TASKS, TASKS, TaskSpec
from .suite import expand_variants, task_names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Validate deterministic generated examples without Torch")
    train = commands.add_parser("train", help="Train reference mini GPT and save checkpoints and reports")
    sweep = commands.add_parser("sweep", help="Paired model/sample/noise grids with complete epoch curves")
    for command in (check, train, sweep):
        command.add_argument("--trainer", choices=(*TASKS, *CONTROL_TASKS, "prefix", "core", "rasp", "rasp-l", "all"), default="all")
        command.add_argument("--variants", choices=("core", "all"), default="all",
                             help="all expands comparison controls; core uses exactly the supplied settings")
        command.add_argument("--prefix-mode", choices=("dyck", "blocks", "dyck2", "crasp", "both"), default="both")
        command.add_argument("--length", type=int, default=64)
        command.add_argument("--min-length", type=int)
        command.add_argument("--symbols", type=int, default=256)
        command.add_argument("--pairs", type=int, default=8)
        command.add_argument("--queries", type=int, default=4)
        command.add_argument("--hops", type=int, default=2)
        command.add_argument("--blocks", type=int, default=3)
        command.add_argument("--query-gap", type=int, default=0)
        command.add_argument("--alpha", type=float, default=0.1)
        command.add_argument("--neutral-fraction", type=float, default=0.25)
        command.add_argument("--max-neutral-gap", type=int)
        command.add_argument("--max-balance", type=int, default=8)
        command.add_argument("--bracket-types", type=int, default=2)
        command.add_argument("--unique", action=argparse.BooleanOptionalAction, default=False)
        command.add_argument("--histogram-bos", action=argparse.BooleanOptionalAction, default=True)
        command.add_argument("--number-limit", type=int, default=512)
        command.add_argument("--scratchpad", choices=("none", "counts", "itemized", "running", "ones"), default="none")
        command.add_argument("--addition-order", choices=("forward", "reverse"), default="forward")
        command.add_argument("--index-hints", action=argparse.BooleanOptionalAction, default=False)
        command.add_argument("--carry-sampling", choices=("standard", "balanced"), default="balanced")
        command.add_argument("--carry-length", type=int)
        command.add_argument("--and-shift", action=argparse.BooleanOptionalAction, default=True)
        command.add_argument("--formula-depth", type=int, default=2)
        command.add_argument("--formula-seed", type=int, default=0)
        command.add_argument("--data-seed", type=int, default=0)
    check.add_argument("--examples", type=int, default=64)
    for command in (train, sweep):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--steps", type=int, default=200)
        command.add_argument("--eval-every", type=int, default=20)
        command.add_argument("--batch-size", type=int, default=16)
        command.add_argument("--train-examples", type=int, default=512)
        command.add_argument("--validation-examples", type=int, default=128)
        command.add_argument("--test-examples", type=int, default=128)
        command.add_argument("--eval-lengths", type=int, nargs="+")
        command.add_argument("--seeds", type=int, nargs="+", default=[0])
        command.add_argument("--width", type=int, default=64)
        command.add_argument("--layers", type=int, default=2)
        command.add_argument("--heads", type=int, default=4)
        command.add_argument("--ff-multiplier", type=int, default=4)
        command.add_argument("--init-std", type=float, help="Normal initialization for matrices; 0.02 matches optimizer benchmark")
        command.add_argument("--optimizer", choices=("adamw", "amsgradw"), default="adamw")
        command.add_argument("--beta1", type=float, default=0.9)
        command.add_argument("--beta2", type=float, default=0.999)
        command.add_argument("--optimizer-epsilon", type=float, default=1e-8)
        command.add_argument("--learning-rate", type=float, default=0.001)
        command.add_argument("--weight-decay", type=float, default=0.01)
        command.add_argument("--grad-clip", type=float, default=1.0)
        command.add_argument("--no-grad-clip", dest="grad_clip", action="store_const", const=None)
        command.add_argument("--device", default="cpu")
        command.add_argument("--cpu-threads", type=int, default=1)
        command.add_argument("--target", type=float, default=0.95)
        command.add_argument("--target-metric", choices=("token_accuracy", "sequence_accuracy", "balanced_accuracy", "final_answer_accuracy"),
                             default="sequence_accuracy")
        command.add_argument("--stop-at-target", action="store_true")
        command.add_argument("--study", choices=("standard", "memorization", "double_descent"), default="standard")
        command.add_argument("--label-noise", type=float, default=0.0)
        command.add_argument("--noise-seed", type=int, default=0)
        command.add_argument("--split-policy", choices=("independent", "disjoint"), default="independent")
        command.add_argument("--fit-epsilon", type=float, default=0.01)
        command.add_argument("--fit-metric", choices=("example_error", "token_error", "loss"), default="example_error")
        command.add_argument("--generalization-patience", type=int, default=2)
        command.add_argument("--curve-tolerance", type=float, default=0.001)
    sweep.set_defaults(study="double_descent", variants="core", trainer="lookup")
    sweep.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128])
    sweep.add_argument("--layer-counts", type=int, nargs="+", default=[2])
    sweep.add_argument("--sample-sizes", type=int, nargs="+", default=[128, 512, 2048])
    sweep.add_argument("--noise-rates", type=float, nargs="+")
    sweep.add_argument("--data-seeds", type=int, nargs="+")
    sweep.add_argument("--epochs", type=int, help="Use equal complete epochs instead of equal update counts")
    args = parser.parse_args(argv)
    try:
        names = task_names(args.trainer, args.prefix_mode)
        fields = ("length", "min_length", "symbols", "pairs", "queries", "hops", "blocks",
                  "query_gap", "alpha", "neutral_fraction", "max_neutral_gap", "max_balance",
                  "bracket_types", "unique", "histogram_bos", "number_limit", "scratchpad",
                  "addition_order", "index_hints", "carry_sampling", "carry_length", "and_shift",
                  "formula_depth", "formula_seed")
        specs = [TaskSpec(task=name, **{field: getattr(args, field) for field in fields}) for name in names]
        if args.variants == "all":
            specs = [variant for spec in specs for variant in expand_variants(spec)]
        if args.command == "check":
            for spec in specs:
                split = build_split(spec, "validation", args.data_seed, args.examples)
                print(json.dumps({"task": spec.task, "variant": spec.run_name, "family": spec.family,
                                  "generative": spec.generative, "context_length": spec.context_length,
                                  "examples": len(split.examples), "fingerprint": split.fingerprint}))
            return
        from .training import train_run

        model_spec = ModelSpec(args.width, args.layers, args.heads, args.ff_multiplier, init_std=args.init_std)
        config = TrainConfig(
            steps=args.steps, batch_size=args.batch_size, eval_every=args.eval_every,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
            data_seed=args.data_seed, train_examples=args.train_examples,
            validation_examples=args.validation_examples, test_examples=args.test_examples,
            eval_lengths=tuple(args.eval_lengths or (2 * args.length, 4 * args.length)),
            device=args.device, cpu_threads=args.cpu_threads, target=args.target,
            target_metric=args.target_metric, stop_at_target=args.stop_at_target,
            study=args.study, label_noise=args.label_noise, noise_seed=args.noise_seed,
            split_policy=args.split_policy, fit_epsilon=args.fit_epsilon, fit_metric=args.fit_metric,
            generalization_patience=args.generalization_patience, curve_tolerance=args.curve_tolerance,
            optimizer=args.optimizer, beta1=args.beta1, beta2=args.beta2, optimizer_epsilon=args.optimizer_epsilon,
        )
        if not args.seeds or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
            raise ValueError("Model seeds must be distinct and nonnegative")
        # Validate the complete matrix before the first training run starts.
        for spec in specs:
            from .studies import validate_study

            validate_study(spec, config)
            for length in config.eval_lengths:
                if length <= spec.length:
                    raise ValueError("OOD evaluation lengths must exceed the training maximum")
                replace(spec, length=length, min_length=None)
            if args.target_metric == "final_answer_accuracy" and not spec.generative:
                raise ValueError("final_answer_accuracy requires a generated-answer trainer")
        from .runtime import write_json

        if args.command == "sweep":
            from .sweeps import SweepConfig, run_sweep

            noise_rates = args.noise_rates if args.noise_rates is not None else (
                [0.0] if args.study == "memorization" else [0.0, 0.2])
            grid = SweepConfig(tuple(args.widths), tuple(args.layer_counts), tuple(args.sample_sizes),
                               tuple(noise_rates), tuple(args.seeds), tuple(args.data_seeds or [args.data_seed]), args.epochs)
            if args.study == "standard":
                raise ValueError("Sweeps require a study profile")
            for width in grid.widths:
                for layers in grid.layer_counts:
                    replace(model_spec, width=width, layers=layers)
            for spec in specs:
                for noise in grid.noise_rates:
                    validate_study(spec, replace(config, label_noise=noise))
            for spec in specs:
                result = run_sweep(spec, model_spec, config, grid, args.output / spec.run_name)
                print(json.dumps({"variant": spec.run_name, "runs": len(result["runs"]),
                                  "report": str(args.output / spec.run_name / "sweep.json")}), flush=True)
            return
        summaries = []
        for spec in specs:
            for seed in args.seeds:
                output = args.output / spec.run_name / f"seed-{seed}"
                def progress(record):
                    print(json.dumps({"task": spec.task, "variant": spec.run_name, "seed": seed, **record}), flush=True)
                result = train_run(spec, model_spec, replace(config, seed=seed), output, progress=progress)
                print(json.dumps({"task": spec.task, "seed": seed, "result": str(output / "result.json"),
                                  "steps_completed": result["steps_completed"],
                                  "target_reached": result["target_reached"]}), flush=True)
                summaries.append({"task": spec.task, "variant": spec.run_name, "seed": seed,
                                  "result": str(output / "result.json"), "test": result["test"],
                                  "time_to_target": result["time_to_target"]})
                write_json(args.output / "suite.json", summaries)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
