"""Replay identical-token cancellation in FP32 and FP64 at fixed RMS epsilon.

Exploratory follow-up to residual_diagnostics.py, not a learning ablation.
The FP64 copy preserves the exact FP32 averaged parameters, rotary tables
and RMS epsilon. Letting nn.RMSNorm select its FP64 epsilon would change the
mathematical model, so every norm receives the original FP32 epsilon first.
"""

import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim import build_optimizer, evaluation_weights
from lab.infrastructure.store import RunDirectory
from centered_replay import apply

HERE = Path(__file__).parent


def evaluate(model, rows):
    inputs, targets = rows[:, :-1], rows[:, -1]
    state = model.embedding_norm(model.token_embedding_table(inputs))
    first = state
    length = inputs.shape[1]
    raw = model.blocks[0].sa(model.blocks[0].ln_1(first),
                             model.rope_cos[:length], model.rope_sin[:length])
    for block in model.blocks:
        state = block(state, model.rope_cos[:length], model.rope_sin[:length])
    logits = model.lm_head(model.ln_f(state))[:, -1]
    loss = F.cross_entropy(logits, targets)
    gradient = torch.autograd.grad(loss, first)[0]
    return logits.detach().double(), raw.detach().double(), gradient.detach().double()


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
                task = build_task(experiment.benchmark, experiment.seeds.data, torch.device("cpu"))
                model = build_model(experiment.model, task.vocab, experiment.seeds.model)
                model.load_state_dict(checkpoint["model"])
                optimizer = build_optimizer(experiment.optimizer, model)
                optimizer.load_state_dict(checkpoint["optimizer"])
                with evaluation_weights(optimizer):
                    model = copy.deepcopy(model).eval()
                for norm in model.modules():
                    if isinstance(norm, nn.RMSNorm) and norm.eps is None:
                        norm.eps = torch.finfo(torch.float32).eps
                high = copy.deepcopy(model).double()
                rows = task.splits["heldout"][:512]
                low_output, low_raw, low_gradient = evaluate(model, rows)
                high_output, high_raw, high_gradient = evaluate(high, rows)
                stable_low, stable_low_raw, stable_low_gradient = evaluate(apply(copy.deepcopy(model)), rows)
                stable_high, stable_high_raw, stable_high_gradient = evaluate(apply(copy.deepcopy(high)), rows)
                identical = (rows[:, :-1] == rows[:, :1]).all(-1)
                strata = {}
                for name, mask in (("identical_input_tokens", identical), ("distinct_input_tokens", ~identical)):
                    if not bool(mask.any()):
                        strata[name] = {"examples": 0}
                        continue
                    stats = {"examples": int(mask.sum()),
                             "max_logit_difference": float((low_output[mask] - high_output[mask]).abs().max()),
                             "changed_predictions": int((low_output[mask].argmax(-1) != high_output[mask].argmax(-1)).sum()),
                             "centered_max_logit_difference": float((stable_low[mask] - stable_high[mask]).abs().max()),
                             "centered_changed_predictions": int((stable_low[mask].argmax(-1) != stable_high[mask].argmax(-1)).sum()),
                             "fp32_centering_changes_predictions": int((low_output[mask].argmax(-1) != stable_low[mask].argmax(-1)).sum()),
                             "fp64_centering_logit_difference": float((high_output[mask] - stable_high[mask]).abs().max())}
                    for precision, output, raw, gradient in (("fp32", low_output, low_raw, low_gradient),
                                                             ("fp64", high_output, high_raw, high_gradient),
                                                             ("centered_fp32", stable_low, stable_low_raw, stable_low_gradient),
                                                             ("centered_fp64", stable_high, stable_high_raw, stable_high_gradient)):
                        selected = gradient[mask]
                        stats[precision] = {
                            "accuracy": float((output[mask].argmax(-1) == rows[mask, -1]).double().mean()),
                            "first_attention_raw_answer_rms": float(raw[mask, -1].square().mean().sqrt()),
                            "ambient_initial_gradient_rms": float(selected.square().mean().sqrt()),
                            "summed_position_gradient_rms": float(selected.sum(1).square().mean().sqrt())}
                    strata[name] = stats
                records.append({"label": label, "checkpoint_sha256": hashlib.sha256(
                    (run.path / "checkpoint.pt").read_bytes()).hexdigest(), "strata": strata})
                print(label, strata["identical_input_tokens"], flush=True)
    result = {"scope": "CPU fixed512-row heldout diagnostic; no training or checkpoint changes.",
              "rms_epsilon_both_precisions": torch.finfo(torch.float32).eps,
              "parameter_and_rope_scope": "Same averaged FP32 numbers exactly converted to FP64.",
              "cases": records}
    (HERE / "precision_diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
