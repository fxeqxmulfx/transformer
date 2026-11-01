"""Replay only closed endpoints with their full Schedule-Free evaluation state."""

import json

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
    assert phase["status"] == "learning_closed"
    assert set(phase["cases"]) == {case["label"] for case in declaration["cases"]}
    torch.set_num_threads(1)
    study = load(HERE)
    cases = []
    for case in declaration["cases"]:
        label = case["label"]
        record = phase["cases"][label]
        spec = study.experiments[label]
        directory = RunDirectory(HERE / "runs" / label)
        assert sha(directory.path / "checkpoint.pt") == record["checkpoint_sha256"]
        task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
        model = build_model(spec.model, task.vocab, spec.seeds.model)
        optimizer = ResidualScheduleFreeAdamW(model.parameters(), **vars(spec.optimizer))
        state = directory.checkpoint("cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        with evaluation_weights(optimizer):
            final = {name: evaluate(task, model, rows, spec.evaluate.batch)
                     for name, rows in task.splits.items()}
            gates = {name: parameter.detach().tolist() for name, parameter in model.named_parameters()
                     if "_gain.weight" in name}
        history = directory.records("history")
        gradients = directory.records("gradients")
        cases.append({"label": label, **record, "step": state["step"], "final_cpu_replay": final,
                      "history": history, "parameters": case["parameters"], "gate_weights": gates,
                      "gradient_count": len(gradients),
                      "gradient_zero_count": sum(row["gradient_l2"] == 0 for row in gradients),
                      "training_seconds": state["training_seconds"],
                      "diagnostic_seconds": state["diagnostic_seconds"], "wall_seconds": state["wall_seconds"]})
        print(label, state["step"], final, flush=True)
    write(HERE / "results.json", {"protocol_sha256": sha(HERE / "protocol.json"),
                                  "phase": phase, "cases": cases})
