"""Read-only diagnostics for optimizer stability investigations.

The Section 4 modular task in Convexifying Transformers is unchanged. These
measurements are additional experimental diagnostics, not paper hyperparameters.
"""

from dataclasses import dataclass
import json

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DiagnosticsConfig:
    every: int = 0
    eval_neighbors: bool = False
    trace_gradients: bool = False

    def __post_init__(self):
        if self.every < 0:
            raise ValueError("Diagnostic cadence must be nonnegative")

    def sample_update(self, step, eval_every):
        return bool(self.every and step % self.every == 0
                    or self.eval_neighbors and step % eval_every in (1, eval_every - 1))

    def probe(self, step, eval_every):
        return self.eval_neighbors and step % eval_every in (1, eval_every - 1)


def append_json(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def truncate_to_checkpoint(path, completed):
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    path.write_text("".join(json.dumps(row) + "\n" for row in rows if row["step"] <= completed))


def prepare_gradient_trace(path, completed):
    """Discard uncheckpointed updates, retaining every completed gradient norm."""
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    rows = [row for row in rows if row["step"] <= completed]
    if [row["step"] for row in rows] != list(range(1, completed + 1)):
        raise ValueError("Checkpointed gradient trace is missing or duplicated")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@torch.no_grad()
def before_update(model, output, targets):
    parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    losses = F.cross_entropy(output.detach().reshape(-1, output.shape[-1]),
                             targets.reshape(-1), reduction="none").reshape(-1, 2).mean(dim=0)
    return parameters, losses.cpu().tolist()


@torch.no_grad()
def after_update(model, optimizer, before, losses, gradient_norm):
    """Read parameter, gradient, actual update and stored moment norms per tensor."""
    tensors, names, keys, temperatures = [], [], [], {}
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter, {})
        values = {
            "parameter_l2": parameter.norm(), "parameter_before_l2": before[name].norm(),
            "gradient_l2": parameter.grad.norm() if parameter.grad is not None else parameter.new_zeros(()),
            "update_l2": (parameter - before[name]).norm(),
        }
        for key in ("m", "v", "maximum", "exp_avg", "exp_avg_sq"):
            if key in state:
                values[key + "_l2"] = state[key].norm()
        if "maximum" in state:
            values["maximum_min"] = state["maximum"].min()
            values["maximum_max"] = state["maximum"].max()
        names.append(name)
        keys.append(list(values))
        tensors.extend(values.values())
        if name.endswith("log_alpha"):
            temperatures[name] = {"log_alpha": parameter.cpu().tolist(),
                                  "inverse_temperature": parameter.exp().cpu().tolist()}
    scalars = iter(torch.stack(tensors).cpu().tolist())
    parameters = {name: {key: next(scalars) for key in fields}
                  for name, fields in zip(names, keys)}
    moment_scope = ("raw_buffers_without_bias_correction_for_AMSGradW; actual_parameter_updates"
                    if any("maximum" in state for state in optimizer.state.values()) else
                    "AdamW_exp_avg_and_exp_avg_sq_before_bias_correction; actual_parameter_updates")
    return {"gradient_l2": float(gradient_norm), "answer_loss": losses[0], "EOS_loss": losses[1],
            "parameters": parameters, "temperatures": temperatures,
            "moment_scope": moment_scope}
