"""Train the author-reference modular-division model and explicit GPTMini controls.

Sources: Convexifying Transformers, arXiv:2211.11052v1, Section 4; the
openai/grok author implementation at the commit recorded in reference-source.json.
This reproduces the arithmetic setting and author model/optimizer, with an
explicit fixed budget and checkpoints rather than test-dependent stopping.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import time

import torch
import torch.nn.functional as F

from experiments.gpt_mini import Config, GPTMini
from experiments.optimizer_benchmark.coordinate import CoordinateOptimizer
from ..curves import curve_witness
from ..runtime import synchronize, write_json
from .modular_data import make_corpus
from .reference_transformer import Transformer
from .batches import next_batch
from .diagnostics import (DiagnosticsConfig, after_update, append_json, before_update,
                          prepare_gradient_trace, truncate_to_checkpoint)
from .provenance import source_hashes


@dataclass(frozen=True)
class RunConfig:
    model: str = "reference"
    optimizer: str = "adamw"
    prime: int = 97
    train_fraction: float = .2
    data_seed: int = 0
    seed: int = 0
    width: int = 128
    layers: int = 2
    heads: int = 4
    steps: int = 150000
    batch_size: int = 512
    batch_policy: str = "short_final"
    eval_every: int = 250
    learning_rate: float = .001
    weight_decay: float = 1.0
    warmup_steps: int = 10
    target: float = .99
    patience: int = 2
    device: str = "cuda"

    def __post_init__(self):
        if self.model not in ("reference", "gptmini") or self.optimizer not in ("adamw", "amsgradw"):
            raise ValueError("Unknown model or optimizer")
        if self.batch_policy not in ("short_final", "wrap_epoch"):
            raise ValueError("Unknown batch policy")
        if min(self.width, self.layers, self.heads, self.steps, self.batch_size, self.eval_every, self.patience) < 1:
            raise ValueError("Dimensions and training counts must be positive")
        if self.width % self.heads or self.width % 2 or (self.model == "gptmini" and self.width // self.heads % 2):
            raise ValueError("Invalid head/position dimensions")
        if self.warmup_steps < 0 or not 0 < self.target <= 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer or event settings")


def learning_rate(config, completed_steps):
    return config.learning_rate * min(1, completed_steps / max(1, config.warmup_steps)) if config.warmup_steps else config.learning_rate


def make_model(config, vocab_size):
    torch.manual_seed(config.seed)
    if config.model == "reference":
        model = Transformer(n_layers=config.layers, n_heads=config.heads, d_model=config.width,
                            dropout=0, max_context_len=50, vocab_len=vocab_size, non_linearity="relu")
    else:
        model = GPTMini(Config(vocab_size=vocab_size, d_model=config.width, n_layers=config.layers,
                               n_heads=config.heads, d_ff=4 * config.width, max_seq_len=50))
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                torch.nn.init.normal_(parameter, std=.02)
    return model.to(config.device, torch.float32)


def logits(model, tokens):
    output = model(tokens)
    return output[0] if isinstance(output, tuple) else output


def make_optimizer(model, config):
    if config.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=0, betas=(.9, .98), eps=1e-8, weight_decay=config.weight_decay)
    return CoordinateOptimizer(model.named_parameters(), 0, rule="amsgrad", beta=.9, beta2=.999,
                               eps=1e-8, decay=config.weight_decay)


@torch.no_grad()
def evaluate(model, rows, batch_size=1024):
    model.eval()
    correct, answers, stops, loss = 0, 0, 0, 0.0
    answer_loss, stop_loss = 0.0, 0.0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        output = logits(model, batch[:, :-1])[:, 4:, :]
        target = batch[:, 5:]
        predicted = output.argmax(dim=-1)
        correct += (predicted == target).all(dim=1).sum().item()
        answers += (predicted[:, 0] == target[:, 0]).sum().item()
        stops += (predicted[:, 1] == target[:, 1]).sum().item()
        loss += F.cross_entropy(output.reshape(-1, output.shape[-1]), target.reshape(-1), reduction="sum").item()
        answer_loss += F.cross_entropy(output[:, 0], target[:, 0], reduction="sum").item()
        stop_loss += F.cross_entropy(output[:, 1], target[:, 1], reduction="sum").item()
    return {"accuracy": correct / len(rows), "answer_accuracy": answers / len(rows),
            "EOS_accuracy": stops / len(rows), "loss": loss / (2 * len(rows)),
            "answer_loss": answer_loss / len(rows), "EOS_loss": stop_loss / len(rows), "examples": len(rows)}


def transition(history, config):
    fit = next((point["step"] for point in history if point["train"]["accuracy"] >= config.target), None)
    confirmed, onset = None, None
    for start in range(len(history) - config.patience + 1):
        streak = history[start:start + config.patience]
        if all(point["heldout"]["accuracy"] >= config.target for point in streak):
            onset, confirmed = streak[0]["step"], streak[-1]["step"]
            break
    return {"train_fit_step": fit, "heldout_onset_step": onset, "heldout_confirmed_step": confirmed,
            "lag_steps": onset - fit if fit is not None and onset is not None else None,
            "delayed_generalization": fit is not None and onset is not None and onset > fit,
            "target": config.target, "patience": config.patience,
            "scope": "two_way_exhaustive_fixed_prime_arithmetic; no_length_transfer_or_causal_claim"}


def train(config, directory, *, resume=False, progress=None, diagnostics=DiagnosticsConfig()):
    torch.set_num_threads(1)
    directory = Path(directory)
    checkpoint_path = directory / "checkpoint.pt"
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise FileExistsError("Output directory must be empty; use resume only for a matching protocol")
    if resume and not checkpoint_path.exists():
        raise FileNotFoundError("No resumable checkpoint")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if config.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(config.device)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    corpus = make_corpus(config.prime, config.train_fraction, config.data_seed)
    model = make_model(config, len(corpus.tokens))
    optimizer = make_optimizer(model, config)
    train_rows = torch.tensor(corpus.train, device=config.device, dtype=torch.long)
    heldout_rows = torch.tensor(corpus.heldout, device=config.device, dtype=torch.long)
    generator = torch.Generator().manual_seed(config.seed + 10000)
    permutation = torch.randperm(len(train_rows), generator=generator)
    cursor, completed, seen, training_seconds, previous_wall = 0, 0, 0, 0.0, 0.0
    history = []
    diagnostic_seconds, segment_diagnostic_seconds, last_batch_size = 0.0, 0.0, None
    plan = {"status": "running", "config": asdict(config), "corpus": corpus.summary(),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "dtype": "float32", "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(config.device) if config.device.startswith("cuda") else None,
            "source_hashes": source_hashes(), "python": platform.python_version(),
            "instrumentation": asdict(diagnostics),
            "batch_policy": config.batch_policy,
            "source_commit": "3d64b1d8c1d595dd8ebdb7771998823f1b14c7b3",
            "source_section": "papers/arXiv-2211.11052v1/arxiv.tex, Section 4, mod-97 experiments",
            "optimizer": {"name": config.optimizer, "betas": [.9, .98 if config.optimizer == "adamw" else .999],
                          "epsilon": 1e-8, "decay_scope": "all_trainable_parameters",
                          "bias_correction": config.optimizer == "adamw", "gradient_clipping": None},
            "deviations": ["fixed_full_budget_instead_of_test_target_stopping", "separate_batch_shuffle_rng",
                           "exact_paper_train_fraction_and_regularization_not_disclosed_in_local_manuscript"],
            "scope": "author_model_reference_protocol" if config.model == "reference" and config.optimizer == "adamw" else "explicit_model/optimizer_adaptation",
            "budget_extensions": []}
    if config.batch_policy == "wrap_epoch":
        plan["deviations"].append("full_batches_across_shuffled_epochs; more_examples_at_fixed_updates")
    if resume:
        saved_plan = json.loads((directory / "plan.json").read_text())
        old_config = saved_plan["config"]
        if any(old_config.get(key) != value for key, value in asdict(config).items() if key != "steps") or config.steps < old_config["steps"]:
            raise ValueError("Resume may only extend the original update budget")
        if saved_plan["source_hashes"] != plan["source_hashes"] or saved_plan.get("instrumentation", asdict(DiagnosticsConfig())) != asdict(diagnostics):
            raise ValueError("Resume sources or instrumentation differ from the frozen plan")
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if isinstance(optimizer, CoordinateOptimizer):
            optimizer.steps = checkpoint["optimizer_steps"]
        completed, seen = checkpoint["step"], checkpoint["examples_seen"]
        training_seconds, previous_wall = checkpoint["training_seconds"], checkpoint["wall_seconds"]
        diagnostic_seconds = checkpoint.get("diagnostic_seconds", 0.0)
        last_batch_size = checkpoint.get("last_batch_size")
        generator.set_state(checkpoint["batch_generator_state"].cpu())
        permutation, cursor = checkpoint["permutation"].cpu(), checkpoint["cursor"]
        history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
        history = [point for point in history if point["step"] <= completed]
        plan["budget_extensions"] = saved_plan["budget_extensions"]
        if config.steps > old_config["steps"]:
            plan["budget_extensions"].append({"old_steps": old_config["steps"], "new_steps": config.steps, "scope": "posthoc_budget_extension"})
    write_json(directory / "plan.json", plan)
    history_path = directory / "history.jsonl"
    history_path.write_text("".join(json.dumps(point) + "\n" for point in history))
    diagnostic_path, probe_path = directory / "diagnostics.jsonl", directory / "probes.jsonl"
    gradient_path = directory / "gradients.jsonl"
    if diagnostics.trace_gradients:
        prepare_gradient_trace(gradient_path, completed)
    if diagnostics.every or diagnostics.eval_neighbors:
        truncate_to_checkpoint(diagnostic_path, completed)
        truncate_to_checkpoint(probe_path, completed)

    def observe(step, *, probe=False):
        row = {"step": step, "epochs_seen": seen / len(train_rows), "training_seconds": training_seconds,
               "wall_seconds": previous_wall + time.perf_counter() - started,
               "last_batch_size": last_batch_size,
               "train": evaluate(model, train_rows), "heldout": evaluate(model, heldout_rows)}
        if probe:
            append_json(probe_path, row)
        else:
            history.append(row)
            append_json(history_path, row)
        if progress:
            progress({"diagnostic_probe": True, **row} if probe else row)

    if not history:
        observe(0)
    synchronize(config.device)
    segment_started = time.perf_counter()
    for step in range(completed + 1, config.steps + 1):
        selected, permutation, cursor, batch_metadata = next_batch(
            permutation, cursor, generator, config.batch_size, config.batch_policy)
        selected = selected.to(config.device)
        batch = train_rows[selected]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(config, step - 1)
        output = logits(model, batch[:, :-1])[:, 4:, :]
        loss = F.cross_entropy(output.reshape(-1, output.shape[-1]), batch[:, 5:].reshape(-1))
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        sampled = diagnostics.sample_update(step, config.eval_every)
        if sampled:
            synchronize(config.device)
            diagnostic_started = time.perf_counter()
            before, component_losses = before_update(model, output, batch[:, 5:])
            segment_diagnostic_seconds += time.perf_counter() - diagnostic_started
        optimizer.step()
        seen += len(batch)
        last_batch_size = len(batch)
        if diagnostics.trace_gradients:
            synchronize(config.device)
            diagnostic_started = time.perf_counter()
            append_json(gradient_path, {"step": step, "batch_size": len(batch),
                "epoch_tail": batch_metadata["epoch_tail"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_l2": float(gradient_norm)})
            segment_diagnostic_seconds += time.perf_counter() - diagnostic_started
        if sampled:
            synchronize(config.device)
            diagnostic_started = time.perf_counter()
            measurements = after_update(model, optimizer, before, component_losses, gradient_norm)
            append_json(diagnostic_path, {"step": step, "batch_size": len(batch),
                **batch_metadata,
                "learning_rate": optimizer.param_groups[0]["lr"], **measurements})
            segment_diagnostic_seconds += time.perf_counter() - diagnostic_started
        canonical = step % config.eval_every == 0 or step == config.steps
        if canonical or diagnostics.probe(step, config.eval_every):
            synchronize(config.device)
            training_seconds += time.perf_counter() - segment_started - segment_diagnostic_seconds
            diagnostic_seconds += segment_diagnostic_seconds
            segment_diagnostic_seconds = 0.0
            observe(step, probe=not canonical)
            if canonical and (step % 5000 == 0 or step == config.steps):
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                    "optimizer_steps": optimizer.steps if isinstance(optimizer, CoordinateOptimizer) else None,
                    "examples_seen": seen, "training_seconds": training_seconds,
                    "diagnostic_seconds": diagnostic_seconds, "last_batch_size": last_batch_size,
                    "wall_seconds": previous_wall + time.perf_counter() - started,
                    "batch_generator_state": generator.get_state(), "permutation": permutation, "cursor": cursor}, checkpoint_path)
            synchronize(config.device)
            segment_started = time.perf_counter()
    plan["status"] = "complete"
    write_json(directory / "plan.json", plan)
    report = {"plan": plan, "completed_steps": config.steps, "training_seconds": training_seconds,
              "diagnostic_seconds": diagnostic_seconds,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(config.device) if config.device.startswith("cuda") else None,
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(config.device) if config.device.startswith("cuda") else None,
              "wall_seconds": previous_wall + time.perf_counter() - started, "transition": transition(history, config),
              "epoch_loss_witness": curve_witness([(point["step"], point["heldout"]["loss"]) for point in history], .02),
              "epoch_error_witness": curve_witness([(point["step"], 1 - point["heldout"]["accuracy"]) for point in history], .02),
              "final": history[-1], "history": history}
    write_json(directory / "measurements.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("reference", "gptmini"), default="reference")
    parser.add_argument("--optimizer", choices=("adamw", "amsgradw"), default="adamw")
    parser.add_argument("--prime", type=int, default=97)
    parser.add_argument("--train-fraction", type=float, default=.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=150000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batch-policy", choices=("short_final", "wrap_epoch"), default="short_final")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--diagnostics-every", type=int, default=0)
    parser.add_argument("--eval-neighbors", action="store_true")
    args = parser.parse_args(argv)
    config = RunConfig(**{key: value for key, value in vars(args).items()
                         if key not in ("output", "resume", "diagnostics_every", "eval_neighbors")})
    report = train(config, args.output, resume=args.resume,
                   diagnostics=DiagnosticsConfig(args.diagnostics_every, args.eval_neighbors),
                   progress=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps({"status": report["plan"]["status"], "transition": report["transition"]}), flush=True)


if __name__ == "__main__":
    main()
