"""Evaluate reserved original-GPTMini endpoints only after the control study closes."""

import json
import math

import torch

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.loop import evaluate
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim import evaluation_weights
from lab.infrastructure.optim.residual_schedule_free import ResidualScheduleFreeAdamW
from lab.infrastructure.store import RunDirectory
from protocol import HERE, frozen, sha, write


def report():
    declaration = frozen()
    phase = json.loads((HERE / "runs/phase.json").read_text())
    if phase["status"] != "learning_closed" or set(phase["cases"]) != {c["label"] for c in declaration["cases"]}:
        raise ValueError("Final test remains sealed until the full control study closes")
    torch.set_num_threads(1)
    study, cases = load(HERE), []
    for case in declaration["cases"]:
        label = case["label"]
        spec, record = study.experiments[label], phase["cases"][label]
        directory = RunDirectory(HERE / "runs" / label)
        if sha(directory.path / "checkpoint.pt") != record["checkpoint_sha256"]:
            raise ValueError("The closed endpoint changed")
        state = directory.checkpoint("cpu")
        task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
        model = build_model(spec.model, task.vocab, spec.seeds.model)
        optimizer = ResidualScheduleFreeAdamW(model.parameters(), **vars(spec.optimizer))
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        with evaluation_weights(optimizer):
            metrics = {name: evaluate(task, model, task.splits[name], spec.evaluate.batch, static=True)
                       for name in ("validation", "test")}
            temperatures = {name: value.detach().exp().tolist()
                            for name, value in model.named_parameters() if name.endswith("log_alpha")}
        history, gradients = directory.records("history"), directory.records("gradients")
        if [r["step"] for r in gradients] != list(range(1, state["step"] + 1)):
            raise ValueError("Incomplete gradient trace")
        if not all(math.isfinite(r["gradient_l2"]) and r["gradient_l2"] > 0 for r in gradients):
            raise ValueError("Gradient trace contains a nonfinite or zero update")
        crossings = [r["step"] for r in history if spec.benchmark.stable([r])]
        if record["status"] == "validation99" and crossings != [state["step"]]:
            raise ValueError("The endpoint is not the first observed crossing")
        for row in history:
            row["validation"].pop("classes", None)
        cases.append({**case, **record, "step": state["step"], "final_cpu_replay": metrics,
                      "history": history, "gradient_count": len(gradients), "all_gradients_finite_nonzero": True,
                      "training_seconds": state["training_seconds"], "wall_seconds": state["wall_seconds"],
                      "attention_temperatures": temperatures,
                      "raw_record_sha256": {f"{n}.jsonl": sha(directory.path / f"{n}.jsonl") for n in ("history", "gradients")}})
        print(label, state["step"], {k: v["sequence_accuracy"] for k, v in metrics.items()}, flush=True)
    write(HERE / "results.json", {"protocol_sha256": sha(HERE / "protocol.json"), "phase": phase, "cases": cases})
