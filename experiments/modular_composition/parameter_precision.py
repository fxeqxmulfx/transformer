"""Audit actual diff_gpt parameter gradients on eight frozen lab checkpoints.

Exploratory numerical follow-up, not a training intervention. Run from python/
with diff_gpt/src on PYTHONPATH. Every variant starts from identical averaged
FP32 weights and rotary tables. FP64 copies retain the original RMS epsilon.
The centered-only failure remains an explicit comparison with the final guarded
prefix implementation; the lab training code and checkpoints are never changed.
"""

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from diff_gpt.domain.model.gpt import GPT
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim import build_optimizer, evaluation_weights
from lab.infrastructure.store import RunDirectory
from centered_replay import apply_guard

HERE = Path(__file__).parent
EPS = torch.finfo(torch.float32).eps
RMS_NORM = F.rms_norm


def fixed_epsilon(input, normalized_shape, weight=None, eps=None):
    """Hold the mathematical model fixed when functional Q/K become FP64."""
    return RMS_NORM(input, normalized_shape, weight, EPS if eps is None else eps)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(model, rows, *, source=True):
    output = model(rows[:, :-1])
    logits = (output[0] if source else output)[:, -1]
    loss = F.cross_entropy(logits, rows[:, -1])
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    return {
        "logits": logits.detach().double(),
        "gradient": torch.cat([g.detach().flatten().double() for g in gradients]),
        "loss": float(loss.detach()),
    }


def compare(result, reference):
    grad, ref = result["gradient"], reference["gradient"]
    return {
        "loss": result["loss"],
        "gradient_rms": float(grad.square().mean().sqrt()),
        "gradient_error_rms": float((grad - ref).square().mean().sqrt()),
        "gradient_error_max": float((grad - ref).abs().max()),
        "gradient_cosine": float(F.cosine_similarity(grad, ref, dim=0)),
        "max_logit_error": float((result["logits"] - reference["logits"]).abs().max()),
        "changed_predictions": int((result["logits"].argmax(-1) !=
                                    reference["logits"].argmax(-1)).sum()),
    }


def main():
    torch.set_num_threads(1)
    study = load(HERE)
    records = []
    for prime in (97, 193):
        for operation in ("division", "mixed"):
            for shape in ("w256d1", "w128d4"):
                label = f"pairs-{operation}-p{prime}-{shape}"
                experiment = study.experiments[label]
                run = RunDirectory(HERE / "runs" / label)
                checkpoint = run.checkpoint("cpu")
                task = build_task(experiment.benchmark, experiment.seeds.data,
                                  torch.device("cpu"))
                lab = build_model(experiment.model, task.vocab, experiment.seeds.model)
                lab.load_state_dict(checkpoint["model"])
                optimizer = build_optimizer(experiment.optimizer, lab)
                optimizer.load_state_dict(checkpoint["optimizer"])
                with evaluation_weights(optimizer):
                    lab = copy.deepcopy(lab).eval()
                for norm in lab.modules():
                    if isinstance(norm, nn.RMSNorm) and norm.eps is None:
                        norm.eps = EPS
                spec = experiment.model
                model = GPT(task.vocab, n_embd=spec.width, n_head=spec.heads,
                            n_layer=spec.depth, block_size=spec.context,
                            use_checkpoint=False, pad_vocab_size_to=1).eval()
                model.load_state_dict(lab.state_dict())
                model.rope_cos.copy_(lab.rope_cos)
                model.rope_sin.copy_(lab.rope_sin)
                for norm in model.modules():
                    if isinstance(norm, nn.RMSNorm) and norm.eps is None:
                        norm.eps = EPS
                rows = task.splits["heldout"][:512]
                outputs = {}
                with patch.object(F, "rms_norm", fixed_epsilon):
                    for variant in ("legacy", "centered", "guarded"):
                        for bits, dtype in ((32, torch.float32), (64, torch.float64)):
                            candidate = copy.deepcopy(model).to(dtype)
                            candidate.centered_xsa = variant == "guarded"
                            for block in candidate.blocks:
                                block.sa.centered_xsa = variant != "legacy"
                            outputs[f"{variant}{bits}"] = evaluate(candidate, rows)
                    lab_result = evaluate(apply_guard(lab), rows, source=False)
                ref = outputs["guarded64"]
                record = {
                    "label": label,
                    "checkpoint_sha256": digest(run.path / "checkpoint.pt"),
                    "rows": len(rows),
                    "repeated_rows": int((rows[:, :-1] == rows[:, :1]).all(-1).sum()),
                    "reference": "actual diff_gpt guarded FP64, fixed FP32 RMS epsilon",
                    "comparisons": {name: compare(value, ref) for name, value in outputs.items()},
                    "lab_guarded32_vs_source_guarded32": compare(lab_result, outputs["guarded32"]),
                }
                records.append(record)
                print(json.dumps(record), flush=True)
    import diff_gpt.domain.model.gpt as source_module
    source = Path(source_module.__file__).parent
    result = {
        "scope": "CPU first512 heldout rows; no training or checkpoint modifications",
        "python": __import__("sys").version,
        "torch": torch.__version__,
        "rms_epsilon_all_variants": EPS,
        "fp64_conversion": "Exact FP32 averaged weights and rotary tables; explicit RMS epsilon",
        "source_sha256": {name: digest(source / name) for name in
                          ("gpt.py", "block.py", "mha.py", "centered_xsa.py")},
        "script_sha256": digest(Path(__file__)),
        "cases": records,
    }
    (HERE / "parameter_precision.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
