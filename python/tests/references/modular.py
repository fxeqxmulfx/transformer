"""Measure the modular trainer at 95f73a9, including its published variants."""

from dataclasses import fields
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F

from common import hashes, records, sha, untimed


def checkpoint(state):
    return {"step": state["step"], "examples_seen": state["examples_seen"],
            "last_batch_size": state["last_batch_size"], "cursor": state["cursor"],
            "permutation": state["permutation"].tolist(), "generator": sha(state["batch_generator_state"]),
            "model": hashes(state["model"]),
            "optimizer": {str(index): {key: sha(value) if value.ndim else float(value)
                                      for key, value in entry.items()}
                          for index, entry in state["optimizer"]["state"].items()}}


def replay(golden):
    from experiments.synthetic_trainers import architecture_training, scheduled_training
    from experiments.synthetic_trainers.paper_reproduction import grokking
    from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
    from experiments.synthetic_trainers.paper_reproduction.modular_data import make_corpus
    from experiments.synthetic_trainers.sparsemax_attention import replace_attention

    for name, expected in golden["models"].items():
        config = grokking.RunConfig(**{key: value for key, value in expected["config"].items()
                                      if key in {f.name for f in fields(grokking.RunConfig)}})
        model = grokking.make_model(config, expected["vocab"])
        if name.endswith("sparsemax"):
            model = replace_attention(model)
        expected["parameters"] = hashes(dict(model.named_parameters()))
        batch = torch.tensor(make_corpus(11, .2, 0).train)
        output = grokking.logits(model, batch[:, :-1])
        loss = F.cross_entropy(output[:, 4:].reshape(-1, output.shape[-1]), batch[:, 5:].reshape(-1))
        loss.backward()
        expected.update(logits=sha(output), loss=loss.item().hex(),
                        gradients=hashes({key: value.grad for key, value in model.named_parameters()}))

    for name, expected in golden["runs"].items():
        settings = expected["config"]
        if name.endswith("sparsemax"):
            trainer, config = architecture_training.train, architecture_training.ArchitectureRunConfig
            # The short fixture uses a constant rate. Its unused annealing
            # interval must still satisfy the original configuration validator.
            settings = {**settings, "anneal_start": 15, "anneal_end": 25}
        elif name.endswith("cosine"):
            trainer, config = scheduled_training.train, scheduled_training.ScheduledRunConfig
        else:
            trainer, config = grokking.train, grokking.RunConfig
        diagnostics = expected["diagnostics"]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            result = trainer(config(**settings), directory,
                             diagnostics=DiagnosticsConfig(diagnostics["every"], diagnostics["neighbors"],
                                                           diagnostics["gradients"]))
            expected.update(history=untimed(result["history"]),
                            probes=untimed(records(directory / "probes.jsonl")),
                            sampled=records(directory / "diagnostics.jsonl"),
                            gradients=records(directory / "gradients.jsonl"), transition=result["transition"],
                            checkpoint=checkpoint(torch.load(directory / "checkpoint.pt", weights_only=False)))
