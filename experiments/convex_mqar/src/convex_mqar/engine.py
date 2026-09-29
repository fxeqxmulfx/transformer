"""Training, validation selection, resumable checkpoints, and progress logs."""

import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .kernels import autocast, loss_for
from .rope import RopeTransformer
from .runtime import write_json


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, data, batch_size, config, convex=False):
    model.eval()
    tokens, positions, labels = data
    correct = torch.zeros((), device=config.device, dtype=torch.long)
    loss_sum = torch.zeros((), device=config.device, dtype=torch.float64)
    synchronize(config.device)
    started = time.perf_counter()
    for start in range(0, len(tokens), batch_size):
        end = start + batch_size
        if convex:
            predictions = model(tokens[start:end], positions[start:end])
        else:
            with autocast(config):
                logits = model(tokens[start:end], positions[start:end])
                loss = F.cross_entropy(logits.flatten(0, 1), labels[start:end].flatten(),
                                       reduction="sum")
            loss_sum += loss.detach().double()
            predictions = logits.argmax(-1)
        correct += (predictions == labels[start:end]).sum()
    synchronize(config.device)
    elapsed = time.perf_counter() - started
    count = labels.numel()
    return {"accuracy": correct.item() / count, "correct": correct.item(),
            "query_count": count, "loss": None if convex else loss_sum.item() / count,
            "seconds": elapsed, "queries_per_second": count / elapsed}


def optimizer_for(model, learning_rate, weight_decay):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=learning_rate, fused=next(model.parameters()).is_cuda)


def train_run(config, length, width, learning_rate, train_data, validation_data,
              directory, status_path, compiled=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    best_path, current_path = directory / "best.pt", directory / "current.pt"
    if result_path.exists():
        return json.loads(result_path.read_text()), best_path
    torch.manual_seed(config.seed)
    model = RopeTransformer(config.vocab, width, config.layers, config.heads,
                            config.mlp_ratio, config.rope_base).to(config.device)
    optimizer = optimizer_for(model, learning_rate, config.weight_decay)
    batch_size = config.batch_size(length, width)
    steps_per_epoch = math.ceil(len(train_data[0]) / batch_size)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = max(1, int(config.warmup_fraction * total_steps))
    next_epoch = 0
    training_seconds = 0.0
    best_validation = {"accuracy": -1.0, "loss": math.inf}
    best_epoch = -1
    execution_segments = []
    if current_path.exists():
        state = torch.load(current_path, map_location=config.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        next_epoch = state["next_epoch"]
        best_validation, best_epoch = state["best_validation"], state["best_epoch"]
        training_seconds = state["training_seconds"]
        execution_segments = state.get("execution_segments", [])
        if not execution_segments and next_epoch:
            execution_segments = [{"first_epoch": 1, "last_epoch": next_epoch,
                                   "compiled_loss": False}]
    objective = loss_for(model, config, compiled)
    if compiled and next_epoch < config.epochs:
        write_json(status_path, {"event": "compilation_pending", "length": length,
                                 "width": width, "learning_rate": learning_rate,
                                 "next_epoch": next_epoch + 1, "compiled_loss": True,
                                 "time_unix": time.time()})
    tokens, positions, labels = train_data
    for epoch in range(next_epoch, config.epochs):
        if not execution_segments or execution_segments[-1]["compiled_loss"] != compiled:
            execution_segments.append({"first_epoch": epoch + 1, "compiled_loss": compiled})
        execution_segments[-1]["last_epoch"] = epoch + 1
        model.train()
        generator = torch.Generator(device=config.device).manual_seed(config.seed + epoch)
        order = torch.randperm(len(tokens), generator=generator, device=config.device)
        loss_sum = torch.zeros((), dtype=torch.float64, device=config.device)
        synchronize(config.device)
        started = time.perf_counter()
        for step, start in enumerate(range(0, len(tokens), batch_size)):
            global_step = epoch * steps_per_epoch + step + 1
            scheduled_lr = learning_rate * min(1.0, global_step / warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = scheduled_lr
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = objective(tokens[indices], positions[indices], labels[indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch + 1}, step {step + 1}")
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().double() * indices.numel()
            if step % 200 == 0:
                elapsed = time.perf_counter() - started
                status = {"event": "training", "length": length, "width": width,
                          "learning_rate": learning_rate, "epoch": epoch + 1,
                          "epochs": config.epochs, "step": step + 1,
                          "compiled_loss": compiled,
                          "steps_per_epoch": steps_per_epoch, "loss": loss.item(),
                          "examples_per_second": (start + indices.numel()) / elapsed,
                          "time_unix": time.time()}
                write_json(status_path, status)
                print(json.dumps(status), flush=True)
        synchronize(config.device)
        training_seconds += time.perf_counter() - started
        validation = evaluate(model, validation_data, batch_size, config)
        improved = (validation["accuracy"], -validation["loss"]) > (
            best_validation["accuracy"], -best_validation["loss"])
        if improved:
            best_validation, best_epoch = validation, epoch + 1
            torch.save(model.state_dict(), best_path)
        epoch_result = {"event": "epoch", "epoch": epoch + 1, "length": length,
                        "width": width, "learning_rate": learning_rate,
                        "train_loss": loss_sum.item() / len(tokens),
                        "validation": validation, "best_epoch": best_epoch,
                        "training_seconds": training_seconds}
        epoch_result["compiled_loss"] = compiled
        with (directory / "epochs.jsonl").open("a") as stream:
            stream.write(json.dumps(epoch_result) + "\n")
        print(json.dumps(epoch_result), flush=True)
        write_json(status_path, epoch_result | {"time_unix": time.time()})
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "next_epoch": epoch + 1, "best_validation": best_validation,
                    "best_epoch": best_epoch, "training_seconds": training_seconds,
                    "execution_segments": execution_segments},
                   current_path.with_suffix(".tmp"))
        current_path.with_suffix(".tmp").replace(current_path)
    result = {"width": width, "length": length, "learning_rate": learning_rate,
              "epochs_completed": config.epochs, "best_epoch": best_epoch,
              "validation": best_validation, "training_seconds": training_seconds,
              "parameters": sum(p.numel() for p in model.parameters()),
              "batch_size": batch_size, "execution_segments": execution_segments}
    write_json(result_path, result)
    del optimizer, model
    if config.device == "cuda":
        torch.cuda.empty_cache()
    return result, best_path
