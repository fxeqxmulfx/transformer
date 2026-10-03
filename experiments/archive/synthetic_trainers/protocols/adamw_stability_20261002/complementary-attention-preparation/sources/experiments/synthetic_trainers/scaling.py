"""Calculate data/model scales, calibrate fit, then run paired larger-model training."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch

from .config import ModelSpec, TrainConfig
from .data import build_split
from .runtime import write_json
from .scaling_calculations import critical_sample_choice, motif_bound, motif_coverage, parameter_count
from .specs import TaskSpec
from .sweeps import SweepConfig, run_sweep


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase", choices=("all", "calibration", "copy"), default="all")
    parser.add_argument("--dd-steps", type=int, default=3000)
    parser.add_argument("--copy-steps", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    bound = motif_bound()
    parity = TaskSpec(task="parity", length=16, symbols=2)
    copy = TaskSpec(task="copy", length=32, min_length=1, symbols=2)
    model = ModelSpec(width=64, layers=2, heads=8, init_std=.02)
    config = TrainConfig(steps=args.dd_steps, batch_size=64, eval_every=500,
                         train_examples=64, validation_examples=128, test_examples=256,
                         optimizer="amsgradw", learning_rate=.0003, weight_decay=.1,
                         grad_clip=None, study="double_descent", data_seed=1, noise_seed=2,
                         label_noise=.2, eval_lengths=(32, 64), device=args.device, curve_tolerance=.02)
    copy_config = replace(config, steps=args.copy_steps, eval_every=1000, label_noise=0,
                          train_examples=bound["rounded_rows"], validation_examples=64, test_examples=128,
                          learning_rate=.0001, eval_lengths=(64, 128))
    calibration_grid = SweepConfig((64,), (2,), (64, 256, 1024, 4096, 16384), (.2,), (0,), (1,))
    copy_grid = SweepConfig((64, 512), (2, 6), (bound["rounded_rows"],), (0,), (0,), (1,))
    plan = {"status": "running", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": args.phase, "device": args.device, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
            "calculated_copy_diversity": bound,
            "model_parameter_counts": [{"width": width, "layers": layers, "heads": model.heads,
                "parameters": parameter_count(copy.vocab_size, width, layers, model.heads)}
                for width in copy_grid.widths for layers in copy_grid.layer_counts],
            "parity_task": asdict(parity), "copy_task": asdict(copy), "model_template": asdict(model),
            "dd_training": asdict(config), "copy_training": asdict(copy_config),
            "calibration_grid": asdict(calibration_grid), "copy_grid": asdict(copy_grid),
            "width_grid": {"widths": [16, 32, 64, 128, 256, 512], "layers": 2,
                           "seeds": args.seeds, "N_rule": "train_only_calibration_geometric_midpoint"},
            "sources": {"dd_EMC": "src/Transformer/DoubleDescent/Section2_EffectiveComplexity.lean",
                        "no_parameter_EMC_identity": "src/Transformer/DoubleDescent/AppendixD_Interpolation.lean",
                        "RASP_layers": "src/Transformer/RASP/Compilation.lean",
                        "rounded_depth_hierarchy": "src/Transformer/CRASP/Transformers.lean",
                        "paper_copy_size": "papers/arXiv-2310.16028v1/appendix.tex, Table 1"},
            "scope": "empirical_scaling; RASP_does_not_fix_embedding_width; no_theorem_guarantees_GPTMini_DD"}
    write_json(args.output / "plan.json", plan)
    started = time.perf_counter()

    def execute(name, spec, training, grid):
        print(json.dumps({"event": "start_phase", "phase": name, "grid": asdict(grid)}), flush=True)
        def progress(row):
            novel = row["validation_novel"]
            print(json.dumps({"phase": name, "step": row["step"], "epochs": row["epochs_seen"],
                              "train_accuracy": row["train"]["sequence_accuracy"],
                              "validation_novel_accuracy": novel["sequence_accuracy"] if novel else None,
                              "ood_accuracy": {key: value["sequence_accuracy"] for key, value in row["validation_ood"].items()},
                              "training_seconds": row["training_seconds"]}), flush=True)
        report = run_sweep(spec, model, training, grid, args.output / name, progress=progress)
        print(json.dumps({"event": "complete_phase", "phase": name, "runs": len(report["runs"]),
                          "frontiers": report["observed_fit_frontiers"]}), flush=True)
        return report

    if args.phase in ("all", "calibration"):
        calibrated = execute("dd-calibration", parity, config, calibration_grid)
        choice = critical_sample_choice(calibrated["runs"])
        plan["sample_choice"] = choice
        write_json(args.output / "plan.json", plan)
        if args.phase == "all":
            grid = SweepConfig((16, 32, 64, 128, 256, 512), (2,),
                               (choice["chosen_sample_size"],), (.2,), tuple(args.seeds), (1,))
            execute("dd-widths", parity, config, grid)
    if args.phase in ("all", "copy"):
        pool = build_split(copy, "train", 1, bound["rounded_rows"])
        plan["actual_copy_motif_coverage"] = motif_coverage(pool)
        write_json(args.output / "plan.json", plan)
        execute("copy-scaling", copy, copy_config, copy_grid)
    plan["status"] = "complete"
    plan["elapsed_seconds"] = time.perf_counter() - started
    plan["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(args.output / "plan.json", plan)


if __name__ == "__main__":
    main()
