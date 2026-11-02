"""Task161: read-only, post-hoc gate diagnostics on the closed task160 models.

The frozen model implementations are checked against task160's source hashes.
Use training rows and validation only. Inference ablations are not training
runs or proofs of the cause of delayed generalization. No saved state changes.
"""

from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.benchmarks.synthetic import generator
from lab.infrastructure.benchmarks.synthetic.rows import Rows
from lab.infrastructure.benchmarks.synthetic.splits import build_split
from lab.infrastructure.benchmarks.synthetic.vocabulary import IGNORE, PAD
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.nn.adaptive_residual import ResidualGain
from lab.infrastructure.optim import evaluation_weights, training_weights
from lab.infrastructure.optim.residual_schedule_free import ResidualScheduleFreeAdamW
from lab.infrastructure.provenance import code

HERE = Path(__file__).resolve().parents[1]
FIELDS = ("loss", "token_accuracy", "sequence_accuracy", "correct", "target_count", "exact", "example_count")
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(values):
    values = torch.cat(values).double()
    derivative_fraction = values * (2 - values)
    assert torch.isfinite(values).all() and ((values >= 0) & (values <= 2)).all()
    return {"count": values.numel(), "mean": values.mean().item(),
            "std": values.std(correction=0).item(),
            "quantiles_0_1_10_50_90_99_100": values.quantile(torch.tensor(
                [0., .01, .1, .5, .9, .99, 1.], dtype=torch.float64)).tolist(),
            "derivative_fraction_mean": derivative_fraction.mean().item(),
            "fraction_below_10pct_maximum_derivative": (derivative_fraction < .1).double().mean().item()}


@torch.no_grad()
def measured(task, model, rows, constants=None):
    collected, masks, handles = defaultdict(lambda: defaultdict(list)), {}, []

    def capture(name):
        def hook(module, inputs, output):
            if constants is not None:
                return torch.full_like(output, constants[name])
            gain = output.squeeze(-1)
            for population, mask in masks.items():
                collected[name][population].append(gain[mask].detach().clone())
        return hook

    for name, module in model.named_modules():
        if isinstance(module, ResidualGain):
            handles.append(module.register_forward_hook(capture(name)))
    model.eval()
    sums = task.accumulator()
    try:
        for start in range(0, len(rows), 64):
            chunk = rows[start:start + 64]
            masks.update(tokens=chunk.rows[:, 0] != PAD, supervised=chunk.rows[:, 1] != IGNORE)
            task.accumulate(model, chunk, sums, static=True)
    finally:
        for handle in handles:
            handle.remove()
    metrics = task.metrics(sums.tolist(), rows)
    return {"metrics": {key: metrics[key] for key in FIELDS},
            "gates": {name: {population: statistics(values) for population, values in populations.items()}
                      for name, populations in collected.items()}}


def main():
    torch.set_num_threads(1)
    declaration = json.loads((HERE / "protocol.json").read_text())
    closed = json.loads((HERE / "results.json").read_text())
    assert closed["phase"]["status"] == "learning_closed"
    assert closed["protocol_sha256"] == sha(HERE / "protocol.json")
    live_sources = code()
    model_sources = ["infrastructure/nn/" + name + ".py" for name in
                     ("adaptive_residual", "residual_transformer", "source_transformer", "rms_backward", "centered_xsa")]
    model_sources += ["infrastructure/optim/residual_schedule_free.py"]
    for name in model_sources:
        assert live_sources[name] == declaration["engine_sources"][name], name
    study, cases = load(HERE), []
    labels = [c["label"] for c in closed["cases"] if "-depth-" in c["label"]]
    for label in labels:
        spec = study.experiments[label]
        source = next(c for c in closed["cases"] if c["label"] == label)
        expected = next(c for c in declaration["cases"] if c["label"] == label)
        path = HERE / "runs" / label / "checkpoint.pt"
        digest = sha(path)
        assert digest == source["checkpoint_sha256"]
        state = torch.load(path, map_location="cpu", weights_only=True)
        task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
        assert task.fingerprints == expected["data"]["fingerprints"]
        model = build_model(spec.model, task.vocab, spec.seeds.model)
        optimizer = ResidualScheduleFreeAdamW(model.parameters(), **vars(spec.optimizer))
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rows = {"training_first512": Rows(task.splits["train"].examples[:512], torch.device("cpu"), generate=False),
                "validation": task.splits["validation"]}
        additional = None
        if "validation/length-128" in task.splits:
            rows["validation/length-128"] = task.splits["validation/length-128"]
        else:
            # Easy depth did not observe this distribution during training.
            extra = build_split(generator(spec.benchmark.probes["length-128"]),
                                "validation", spec.seeds.data, spec.benchmark.validation)
            rows["validation/length-128"] = Rows(extra.examples, torch.device("cpu"), generate=False)
            additional = {"name": "validation/length-128", "fingerprint": extra.fingerprint,
                          "seed": spec.seeds.data, "count": len(extra.examples)}
        record = {"label": label, "step": state["step"], "checkpoint_sha256": digest,
                  "width": spec.model.width, "depth": spec.model.depth,
                  "additional_validation": additional}
        with sdpa_kernel(SDPBackend.MATH):
            training_weights(optimizer)
            record["working"] = {name: measured(task, model, data) for name, data in rows.items()}
            with evaluation_weights(optimizer):
                record["averaged"] = {name: measured(task, model, data) for name, data in rows.items()}
                for split in (name for name in rows if name in source["final_cpu_replay"]):
                    for field in ("correct", "target_count", "exact", "example_count"):
                        assert record["averaged"][split]["metrics"][field] == source["final_cpu_replay"][split][field]
                gates = record["averaged"]["training_first512"]["gates"]
                if gates:
                    before = copy.deepcopy(model.state_dict())
                    means = {name: values["tokens"]["mean"] for name, values in gates.items()}
                    record["inference_ablations"] = {
                        mode: {name: measured(task, model, data, constants)["metrics"]
                               for name, data in rows.items() if name != "training_first512"}
                        for mode, constants in (("unit", {name: 1. for name in gates}), ("training_token_mean", means))}
                    torch.testing.assert_close(before, model.state_dict(), rtol=0, atol=0)
                record["gate_weight_l2"] = {name: value.norm().item() for name, value in model.named_parameters()
                                             if name.endswith(("attn_gain.weight", "ffn_gain.weight"))}
        assert sha(path) == digest
        history = source["history"]
        record["first99_short_validation"] = next((r["step"] for r in history if r["validation"]["sequence_accuracy"] >= .99), None)
        record["first99_length128_validation"] = next((r["step"] for r in history if r.get("validation/length-128", {}).get("sequence_accuracy", 0) >= .99), None)
        cases.append(record)
        print(label, {k: v["metrics"]["sequence_accuracy"] for k, v in record["averaged"].items()}, flush=True)
    result = {"scope": "Exploratory closed-endpoint diagnostics; only first512 training rows and validation. Easy-depth length128 validation is newly generated with the original problem, seed and validation stream, and fingerprinted separately. No final-test predictions or training updates. Inference ablations are not trained repairs.",
              "limitations": "Depth endpoints have unequal update counts; intermediate weights and per-parameter historical gradients were not retained. Size also changes task and depth. Endpoint associations cannot identify a cause of delayed learning or a width-only optimizer effect.",
              "saturation": "gain=2sigmoid(z); derivative is gain*(2-gain)/2. Fraction below10% compares with its maximum1/2.",
              "source_sha256": sha(Path(__file__)), "protocol_sha256": sha(HERE / "protocol.json"),
              "closed_result_sha256": sha(HERE / "results.json"), "engine_sources": live_sources,
              "checked_frozen_model_sources": model_sources, "cases": cases}
    folder = HERE / "runs/gain_transfer"
    folder.mkdir(exist_ok=True)
    output = folder / "probe.json"
    if output.exists():
        assert json.loads(output.read_text()) == result
    else:
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Complete probe sha256", sha(output), flush=True)


if __name__ == "__main__":
    main()
