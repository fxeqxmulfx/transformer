"""Train mini GPT, select on validation, and evaluate held-out splits afterward."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch

from experiments.gpt_mini import GPTMini
from .data import build_split
from .metrics import evaluate, masked_loss, validation_rank
from .records import collate
from .runtime import optimizer_description, optimizer_for, source_hashes, synchronize, write_json
from .vocabulary import IGNORE


def train_run(spec, model_spec, config, output, model_factory=None, progress=None, *, prepared_splits=None):
    """A factory accepts the reference GPTMini Config and returns a causal module.

    Dataset seeds stay fixed across model seeds. Timing includes synchronization;
    time to target is the first scheduled validation observation at that quality.
    Existing run directories are never overwritten. No test score selects a model.
    """
    started = time.perf_counter()
    from .studies import StudySession, prepare_study, validate_study

    validate_study(spec, config)
    if prepared_splits is not None and config.study == "standard":
        raise ValueError("Prepared pools require a study profile")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    eval_specs = {f"length-{length}": replace(spec, length=length, min_length=None)
                  for length in config.eval_lengths}
    if any(length <= spec.length for length in config.eval_lengths):
        raise ValueError("OOD evaluation lengths must exceed the training maximum")
    if config.target_metric == "final_answer_accuracy" and not spec.generative:
        raise ValueError("final_answer_accuracy is available for generated answers only")
    if spec.task == "boolean_and" and spec.and_shift:
        eval_specs["position_shift"] = replace(spec, and_region="late")
        for length in config.eval_lengths:
            eval_specs[f"position-shift-length-{length}"] = replace(spec, length=length, min_length=None, and_region="late")
    if spec.task == "addition":
        for length in (spec.length, *config.eval_lengths):
            eval_specs[f"hard-carry-length-{length}"] = replace(spec, length=length, min_length=None, carry_length=length)
    directory = Path(output)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Run directory is not empty; choose a fresh output directory")
    directory.mkdir(parents=True, exist_ok=True)
    if device.type == "cpu":
        torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    factory = model_factory or GPTMini
    maximum = max(item.context_length for item in (spec, *eval_specs.values()))
    vocab_size = max(item.vocab_size for item in (spec, *eval_specs.values()))
    model = factory(model_spec.reference_config(vocab_size, maximum))
    if model_spec.init_std is not None:
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim == 2:
                    torch.nn.init.normal_(parameter, mean=0.0, std=model_spec.init_std)
    model = model.to(device=device, dtype=torch.float32)
    optimizer = optimizer_for(model, config)
    study = None
    if config.study != "standard":
        pool, train, noise = prepare_study(spec, config, prepared_splits)
        validation = pool["validation"]
        study = StudySession(spec, config, pool, noise, train, eval_specs, model)
        study.save(directory)
    else:
        train = build_split(spec, "train", config.data_seed, config.train_examples)
        validation = build_split(spec, "validation", config.data_seed, config.validation_examples)
    train.save(directory / "data" / "train")
    validation.save(directory / "data" / "validation")
    provenance = {"task": asdict(spec), "model": asdict(model_spec), "training": asdict(config),
                  "factory": f"{factory.__module__}.{factory.__qualname__}",
                  "torch_version": torch.__version__, "source_hashes": source_hashes(factory, optimizer),
                  "optimizer": optimizer_description(config)}
    provenance["vocab_size"] = vocab_size
    provenance["context_length"] = maximum
    provenance["parameter_storage_bits"] = sum(p.numel() * p.element_size() * 8 for p in model.parameters())
    provenance["parameter_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
    if study:
        provenance["noise"] = noise
    provenance["evaluation_specs"] = {name: asdict(item) for name, item in eval_specs.items()}
    if spec.task == "crasp":
        from .crasp import program_with_witnesses

        formula, witnesses = program_with_witnesses(spec.formula_depth, spec.formula_seed)
        provenance["program"] = {"ast": asdict(formula), "expression": formula.describe(),
                                 "count_depth": formula.depth, "witnesses": witnesses}
    write_json(directory / "config.json", provenance)
    synchronize(device)
    setup_seconds = time.perf_counter() - started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rng = torch.Generator().manual_seed(config.seed)
    order, cursor = [], 0
    training_seconds = 0.0
    examples_seen = supervised_tokens = padded_tokens = completed = 0
    best, best_step, hit, final = None, 0, None, None

    def observe(step):
        nonlocal best, best_step, hit, final
        final = evaluate(model, validation.examples, config.batch_size, device, spec)
        synchronize(device)
        record = {"step": step, "training_seconds": training_seconds,
                  "wall_seconds": time.perf_counter() - started,
                  "examples_seen": examples_seen, "supervised_tokens_seen": supervised_tokens,
                  "validation": final}
        if study:
            study.observe(model, record, device)
            synchronize(device)
            record["wall_seconds"] = time.perf_counter() - started
        selection = "final_answer_accuracy" if config.target_metric == "final_answer_accuracy" else "sequence_accuracy"
        if best is None or validation_rank(final, selection) > validation_rank(best, selection):
            best, best_step = final, step
            torch.save(model.state_dict(), directory / "best.pt")
        if hit is None and config.target is not None and final[config.target_metric] >= config.target:
            hit = {key: record[key] for key in ("step", "training_seconds", "wall_seconds", "examples_seen", "supervised_tokens_seen")}
            hit["value"] = final[config.target_metric]
        with (directory / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if progress:
            progress(record)

    observe(0)
    for step in range(1, config.steps + 1):
        if config.stop_at_target and hit is not None:
            break
        synchronize(device)
        step_started = time.perf_counter()
        if cursor == len(order):
            order = torch.randperm(len(train.examples), generator=rng).tolist()
            cursor = 0
        indices = order[cursor:cursor + config.batch_size]
        cursor += len(indices)
        batch = collate([train.examples[index] for index in indices], device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = masked_loss(model(batch.tokens), batch.targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip if config.grad_clip is not None else float("inf"),
                                       error_if_nonfinite=True)
        optimizer.step()
        synchronize(device)
        training_seconds += time.perf_counter() - step_started
        completed = step
        examples_seen += len(indices)
        supervised_tokens += int((batch.targets != IGNORE).sum())
        padded_tokens += batch.tokens.numel()
        if step % config.eval_every == 0 or step == config.steps:
            observe(step)

    torch.save(model.state_dict(), directory / "final.pt")
    final_training_wall = time.perf_counter() - started
    final_diagnostics = study.checkpoint_diagnostics(model, device) if study else None
    tests, fingerprints = {}, {"train": train.fingerprint, "validation": validation.fingerprint}
    held_out_splits, final_tests = {}, {}
    if study:
        fingerprints["train_clean"] = study.pool["train"].fingerprint
    for name, test_spec in {"in_distribution": spec, **eval_specs}.items():
        held_out = study.pool["test"] if study and name == "in_distribution" else build_split(test_spec, "test", config.data_seed, config.test_examples)
        held_out.save(directory / "data" / name)
        fingerprints[name] = held_out.fingerprint
        held_out_splits[name] = held_out
        if study:
            final_tests[name] = evaluate(model, held_out.examples, config.batch_size, device, test_spec)
    model.load_state_dict(torch.load(directory / "best.pt", map_location=device, weights_only=True))
    for name, held_out in held_out_splits.items():
        tests[name] = evaluate(model, held_out.examples, config.batch_size, device, held_out.spec)
    synchronize(device)
    result = {
        "task": spec.task, "family": spec.family, "provenance": provenance,
        "variant": spec.run_name,
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps_completed": completed, "examples_seen": examples_seen,
        "supervised_tokens_seen": supervised_tokens, "padded_tokens_processed": padded_tokens,
        "setup_seconds": setup_seconds, "training_seconds": training_seconds,
        "training_wall_seconds": final_training_wall, "total_wall_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "target": config.target, "target_metric": config.target_metric,
        "target_reached": hit is not None, "time_to_target": hit,
        "best_step": best_step, "validation_best": best, "validation_final": final,
        "test": tests, "split_fingerprints": fingerprints,
    }
    if study:
        result["test_final"] = final_tests
        result["study"] = study.report()
        result["study"]["final_checkpoint"] = final_diagnostics
        result["study"]["best_checkpoint"] = study.checkpoint_diagnostics(model, device)
        result["study"]["validation_probe_fingerprints"] = {name: split.fingerprint for name, split in study.probes.items()}
        synchronize(device)
        result["peak_cuda_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        result["total_wall_seconds"] = time.perf_counter() - started
    write_json(directory / "result.json", result)
    return result
