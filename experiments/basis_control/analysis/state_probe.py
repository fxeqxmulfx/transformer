"""Compare training and validation at both Schedule-Free weight states.

Task162, adapted from task160 at23154349. Copy an atomic checkpoint
before evaluating it on CPU. It reads the first512 training rows and the full
validation split, never final test. Training accuracy is a subset measurement.
Neither the active checkpoint nor the learning process is modified.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.benchmarks.synthetic.rows import Rows
from lab.infrastructure.engine.loop import evaluate
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim import evaluation_weights, training_weights
from lab.infrastructure.optim.residual_schedule_free import ResidualScheduleFreeAdamW

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from protocol import frozen, sha, write

METRICS = ("loss", "token_accuracy", "sequence_accuracy", "correct", "target_count",
           "exact", "example_count")


def probe(label, snapshot=None):
    declaration = frozen()
    case = next(case for case in declaration["cases"] if case["label"] == label)
    spec = load(HERE).experiments[label]
    if "-recall-" not in label:
        raise ValueError("This training-subset probe is declared only for recall")
    raw = (snapshot or HERE / "runs" / label / "checkpoint.pt").read_bytes()
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    digest = hashlib.sha256(raw).hexdigest()
    folder = HERE / "runs/probes"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{label}-step{state['step']}"
    copied = folder / f"{stem}.pt"
    if copied.exists() and sha(copied) != digest:
        raise ValueError("A different checkpoint already occupies this snapshot")
    if not copied.exists():
        copied.write_bytes(raw)
    torch.set_num_threads(1)
    task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
    if task.fingerprints != case["fingerprints"]:
        raise ValueError("Frozen data changed")
    model = build_model(spec.model, task.vocab, spec.seeds.model)
    optimizer = ResidualScheduleFreeAdamW(model.parameters(), **vars(spec.optimizer))
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    rows = {"first512_training_examples": Rows(task.splits["train"].examples[:512],
                                             torch.device("cpu"), generate=False),
            "validation": task.splits["validation"]}
    record = {"label": label, "step": state["step"], "checkpoint_sha256": digest,
              "protocol_sha256": sha(HERE / "protocol.json"),
              "probe_source_sha256": sha(Path(__file__)),
              "scope": "Exploratory CPU replay; first512 training rows and all512 validation rows; no final test, no training mutation."}
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        training_weights(optimizer)
        temperatures = lambda: {name: value.detach().exp().tolist() for name, value in model.named_parameters()
                                if name.endswith("log_alpha")}
        record["attention_temperatures"] = {"working": temperatures()}
        record["working_weights"] = {name: evaluate(task, model, data, 64, static=True)
                                     for name, data in rows.items()}
        with evaluation_weights(optimizer):
            record["attention_temperatures"]["averaged"] = temperatures()
            record["averaged_weights"] = {name: evaluate(task, model, data, 64, static=True)
                                          for name, data in rows.items()}
    complete = folder / f"{stem}.full.json"
    write(complete, record)
    record["full_result_sha256"] = sha(complete)
    for mode in ("working_weights", "averaged_weights"):
        record[mode] = {name: {key: scores[key] for key in METRICS}
                        for name, scores in record[mode].items()}
    write(folder / f"{stem}.summary.json", record)
    print(json.dumps(record, indent=2), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--snapshot", type=Path)
    options = parser.parse_args()
    probe(options.label, options.snapshot)
