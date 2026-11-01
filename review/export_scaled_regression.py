"""Export both frozen scaled-residual endpoints for actual diff_gpt regressions.

The weights come from the closed lab cohort, and labels come from modular
division. Expected scores come from its unchanged CPU scorer, not from the GPT
implementation under test. This preserves learned held-out behavior; it does
not turn one frozen trajectory into a guarantee about future training speed.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim.residual_schedule_free import ResidualScheduleFreeAdamW
from lab.infrastructure.store import RunDirectory

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/residual_generalization"


@torch.no_grad()
def predictions(model, rows):
    return torch.cat([model(part[:, :-1])[0][:, -1].argmax(-1) for part in rows.split(512)])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = json.loads((HERE / "results.json").read_text())
    if report["phase"]["status"] != "closed":
        raise ValueError("Export only the closed, replayed cohort")
    source = Path(sys.argv[1]).resolve()
    pinned = json.loads((ROOT / "python/tests/fixtures/residual_transformer.json").read_text())
    if any(sha(source / name) != digest for name, digest in pinned["source_files"].items()):
        raise ValueError("GPT model source differs from the independently checked source")
    sys.path.insert(0, str(source / "src"))
    from diff_gpt.domain.model.residual_gpt import ResidualGPT
    destination = source / "tests/fixtures/residual_generalization"
    destination.mkdir(parents=True, exist_ok=True)
    study = load(HERE)
    torch.set_num_threads(1)
    cases = []
    for seed in (11, 17):
        label = f"mechanism-s{seed}-scaled"
        measured = next(c for c in report["cases"] if c["label"] == label)
        spec = study.experiments[label]
        task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
        run = RunDirectory(HERE / "runs" / label)
        if sha(run.path / "checkpoint.pt") != measured["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint changed")
        checkpoint = run.checkpoint("cpu")
        model = build_model(spec.model, task.vocab, spec.seeds.model)
        optimizer = ResidualScheduleFreeAdamW(model.parameters(), **vars(spec.optimizer))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer.eval()
        configuration = {"vocab_size": task.vocab, "n_embd": spec.model.width,
            "block_size": spec.model.context, "n_head": spec.model.heads,
            "n_layer": spec.model.depth, "mixing": "scaled", "use_checkpoint": False,
            "pad_vocab_size_to": 1, "rope_base": spec.model.rope_base}
        actual = ResidualGPT(**configuration).eval()
        actual.load_state_dict(model.state_dict())
        rows = task.splits["test"]
        prediction = predictions(actual, rows)
        digest = hashlib.sha256(prediction.numpy().astype("<i8").tobytes()).hexdigest()
        if digest != measured["final_cpu_fp32"]["test"]["prediction_sha256"]:
            raise AssertionError("Actual GPT does not reproduce the frozen lab predictions")
        correct = int((prediction == rows[:, -1]).sum())
        actual.branch_scale = 1.0
        broken_correct = int((predictions(actual, rows) == rows[:, -1]).sum())
        file = destination / f"scaled_seed{seed}.npz"
        np.savez_compressed(file, **{name: value.detach().numpy() for name, value in model.state_dict().items()},
                            test_rows=rows.numpy().astype("uint8"),
                            train_inputs=task.splits["train"][:, :-1].numpy().astype("uint8"))
        cases.append({"label": label, "file": file.name, "sha256": sha(file),
                      "configuration": configuration, "updates": measured["step"],
                      "source_checkpoint_sha256": measured["checkpoint_sha256"],
                      "test_examples": len(rows), "test_correct": correct,
                      "disabled_scale_correct": broken_correct, "minimum_accuracy": .95,
                      "prediction_sha256": digest})
        print(json.dumps(cases[-1]), flush=True)
    manifest = {"training_commit": "e4edb9f5", "gpt_model_commit": pinned["source_commit"],
                "protocol_sha256": report["protocol_sha256"], "torch": torch.__version__,
                "scope": __doc__, "cases": cases}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
