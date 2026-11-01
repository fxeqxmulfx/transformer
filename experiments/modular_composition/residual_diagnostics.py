"""Inspect residual geometry and gradients without updating saved models.

Source: SourceBlock at33e412cb, Transformer.Residual.residualRun_unroll,
and review/residual_connections.md. Fixed first512 rows of each split are
diagnostic batches, not selected examples or new learning endpoints. The
manual unroll must reproduce the model's logits. Quantities describe the
answer position unless explicitly named all-position gradients. They do
not establish that changing residuals would improve training.
"""

import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim import build_optimizer, evaluation_weights
from lab.infrastructure.store import RunDirectory

HERE = Path(__file__).parent


def rms(value):
    return value.detach().double().square().mean(-1).sqrt()


def batch_diagnostics(model, rows):
    inputs, targets = rows[:, :-1], rows[:, -1]
    length = inputs.shape[1]
    x = model.embedding_norm(model.token_embedding_table(inputs))
    states, branches, raw_outputs, labels = [x], [], [], []
    for index, block in enumerate(model.blocks):
        raw = block.sa(block.ln_1(x), model.rope_cos[:length], model.rope_sin[:length])
        raw_outputs.append(raw)
        branch = block.attn_out_norm(raw)
        branches.append(branch)
        labels.append(f"block{index + 1}.attention")
        x = x + branch
        states.append(x)
        raw = block.ffwd(block.ln_2(x))
        raw_outputs.append(raw)
        branch = block.ffn_out_norm(raw)
        branches.append(branch)
        labels.append(f"block{index + 1}.ffn")
        x = x + branch
        states.append(x)
    logits = model.lm_head(model.ln_f(x))[:, -1]
    with torch.no_grad():
        reference = model(inputs)[:, -1]
        torch.testing.assert_close(logits, reference, rtol=0, atol=0)
        prefix = [float((model.lm_head(model.ln_f(state))[:, -1].argmax(-1) == targets).double().mean())
                  for state in states]
    loss = F.cross_entropy(logits, targets)
    state_gradients = torch.autograd.grad(loss, states, retain_graph=True)
    parameter_gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    parameters = [{"name": name, "parameter_rms": float(rms(parameter.flatten())),
                   "gradient_rms": float(rms(gradient.flatten()))}
                  for (name, parameter), gradient in zip(model.named_parameters(), parameter_gradients, strict=True)]
    geometry = []
    for index, (label, branch) in enumerate(zip(labels, branches, strict=True)):
        before, after = states[index][:, -1], states[index + 1][:, -1]
        delta = branch[:, -1]
        ratio = rms(delta) / rms(before).clamp_min(1e-30)
        geometry.append({"branch": label, "residual_rms_before": float(rms(before).mean()),
                         "branch_rms": float(rms(delta).mean()),
                         "branch_to_residual_rms_mean": float(ratio.mean()),
                         "branch_to_residual_rms_min": float(ratio.min()),
                         "branch_to_residual_rms_max": float(ratio.max()),
                         "cosine_with_residual_mean": float(F.cosine_similarity(before, delta, dim=-1).detach().double().mean()),
                         "norm_shrink_fraction": float((rms(after) < rms(before)).double().mean()),
                         "residual_rms_after": float(rms(after).mean()),
                         "raw_output_rms_mean_by_position": rms(raw_outputs[index]).mean(0).tolist(),
                         "raw_output_rms_min_by_position": rms(raw_outputs[index]).amin(0).tolist()})
    identical = (inputs == inputs[:, :1]).all(-1)
    strata = {}
    for name, mask in (("identical_input_tokens", identical), ("distinct_input_tokens", ~identical)):
        count = int(mask.sum())
        if not count:
            strata[name] = {"examples": 0}
            continue
        gradient = state_gradients[0][mask].detach().double()
        # A repeated token has one shared embedding, so its position gradients add.
        # This is before differentiating embedding RMSNorm and the tied readout.
        summed = gradient.sum(1)
        denominator = gradient.norm(dim=-1).sum(-1).clamp_min(1e-30)
        strata[name] = {"examples": count,
                       "ambient_initial_state_gradient_rms": float(gradient.square().mean().sqrt()),
                       "position_sum_gradient_rms": float(summed.square().mean().sqrt()),
                       "position_sum_cancellation_ratio_mean": float((summed.norm(dim=-1) / denominator).mean()),
                       "first_attention_raw_answer_rms_mean": float(rms(raw_outputs[0][mask, -1]).mean())}
    return {"examples": len(rows), "loss": float(loss.detach()), "accuracy": prefix[-1],
            "prefix_tied_readout_accuracy": prefix,
            "prefix_scope": "Diagnostic early exits using the existing final norm/readout, not separately trained heads.",
            "state_gradient_rms_all_positions": [float(rms(g.flatten())) for g in state_gradients],
            "state_gradient_rms_mean_by_position": [rms(g).mean(0).tolist() for g in state_gradients],
            "state_rms_mean_by_position": [rms(state).mean(0).tolist() for state in states],
            "state_rms_min_by_position": [rms(state).amin(0).tolist() for state in states],
            "branches": geometry, "parameters": parameters, "manual_forward_bitwise_equal": True,
            "input_strata": strata,
            "gradient_scope": "Gradients of the same full512-row mean loss; strata do not renormalize it. Position sums are not the complete tied-parameter gradient."}


def main():
    torch.set_num_threads(1)
    phase = sys.argv[1] if len(sys.argv) > 1 else "early"
    if phase not in ("early", "long"):
        raise ValueError("Select early or long")
    directory = HERE if phase == "early" else HERE.with_name("modular_composition_long")
    study = load(directory)
    records = []
    for prime in (97, 193):
        for operation in ("division", "mixed"):
            for shape in ("w256d1", "w128d4"):
                label = f"pairs-{operation}-p{prime}-{shape}"
                experiment = study.experiments[label]
                run = RunDirectory(directory / "runs" / label)
                checkpoint = run.checkpoint("cpu")
                if checkpoint is None:
                    records.append({"label": label, "status": "no_checkpoint"})
                    continue
                task = build_task(experiment.benchmark, experiment.seeds.data, torch.device("cpu"))
                model = build_model(experiment.model, task.vocab, experiment.seeds.model)
                model.load_state_dict(checkpoint["model"])
                model.eval()
                optimizer = build_optimizer(experiment.optimizer, model)
                optimizer.load_state_dict(checkpoint["optimizer"])
                with evaluation_weights(optimizer):
                    batches = {name: batch_diagnostics(model, task.splits[name][:512]) for name in ("train", "heldout")}
                records.append({"label": label, "status": "measured", "step": checkpoint["step"],
                                "checkpoint_sha256": hashlib.sha256((run.path / "checkpoint.pt").read_bytes()).hexdigest(),
                                "task": task.summary(), "batches": batches})
                print(label, checkpoint["step"], {name: value["accuracy"] for name, value in batches.items()}, flush=True)
    result = {"scope": "Fixed-batch geometry and gradients; correlation, not a causal residual-architecture comparison.",
              "phase": phase, "cases": records}
    (directory / "residual_diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
