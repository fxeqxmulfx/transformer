"""Reserved sampled and novel-input tests from each complete Schedule-Free endpoint."""

import json

import torch

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.benchmarks.synthetic.rows import Rows
from lab.infrastructure.engine.loop import evaluate
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim import evaluation_weights
from lab.infrastructure.optim.residual_schedule_free import ResidualScheduleFreeAdamW
from lab.infrastructure.store import RunDirectory
from protocol import HERE, frozen, novel_examples, sha, write


def report():
    declaration = frozen()
    phase = json.loads((HERE / "runs/phase.json").read_text())
    if phase["status"] != "learning_closed" or set(phase["cases"]) != {c["label"] for c in declaration["cases"]}:
        raise ValueError("Final test is sealed until learning closes")
    torch.set_num_threads(1)
    study, cases = load(HERE), []
    for case in declaration["cases"]:
        label, record = case["label"], phase["cases"][case["label"]]
        spec = study.experiments[label]
        directory = RunDirectory(HERE / "runs" / label)
        if sha(directory.path / "checkpoint.pt") != record["checkpoint_sha256"]:
            raise ValueError("Closed endpoint changed")
        task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
        model = build_model(spec.model, task.vocab, spec.seeds.model)
        optimizer = ResidualScheduleFreeAdamW(model.parameters(), **vars(spec.optimizer))
        state = directory.checkpoint("cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        final = {}
        with evaluation_weights(optimizer):
            for name, rows in task.splits.items():
                if name == "train":
                    continue
                final[name] = evaluate(task, model, rows, spec.evaluate.batch, static=True)
                novel = novel_examples(task, rows)
                final[name + "/novel"] = None if not novel else evaluate(
                    task, model, Rows(novel, torch.device("cpu")), spec.evaluate.batch, static=True)
        history, gradients = directory.records("history"), directory.records("gradients")
        cases.append({"label": label, **record, "step": state["step"], "final_cpu_replay": final,
                      "history": history, "parameters": case["parameters"],
                      "gradient_count": len(gradients),
                      "gradient_zero_count": sum(row["gradient_l2"] == 0 for row in gradients),
                      "training_seconds": state["training_seconds"],
                      "diagnostic_seconds": state["diagnostic_seconds"], "wall_seconds": state["wall_seconds"]})
        print(label, state["step"], {key: None if value is None else value["sequence_accuracy"]
                                     for key, value in final.items()}, flush=True)
    write(HERE / "results.json", {"protocol_sha256": sha(HERE / "protocol.json"), "phase": phase, "cases": cases})
