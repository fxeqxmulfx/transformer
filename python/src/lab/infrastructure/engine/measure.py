"""Per-tensor measurements around one update; none of them changes the trajectory.

Ports of `paper_reproduction.diagnostics.before_update` and `after_update`
for AdamW, whose moments are read as stored, before bias correction.
"""

import torch

MOMENT_SCOPE = "AdamW_exp_avg_and_exp_avg_sq_before_bias_correction; actual_parameter_updates"


@torch.no_grad()
def before_update(model, task, output, targets):
    """Copies of the parameters, and the batch's mean loss at each supervised position."""
    parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    return parameters, task.position_losses(output.detach(), targets).cpu().tolist()


@torch.no_grad()
def after_update(model, optimizer, task, before, losses, gradient_norm):
    """Norms of each parameter, its gradient, its actual update and its moments."""
    tensors, names, keys, temperatures = [], [], [], {}
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter, {})
        values = {
            "parameter_l2": parameter.norm(), "parameter_before_l2": before[name].norm(),
            "gradient_l2": parameter.grad.norm() if parameter.grad is not None else parameter.new_zeros(()),
            "update_l2": (parameter - before[name]).norm(),
        }
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                values[key + "_l2"] = state[key].norm()
        names.append(name)
        keys.append(list(values))
        tensors.extend(values.values())
        if name.endswith("log_alpha"):
            temperatures[name] = {"log_alpha": parameter.cpu().tolist(),
                                  "inverse_temperature": parameter.exp().cpu().tolist()}
    scalars = iter(torch.stack(tensors).cpu().tolist())
    parameters = {name: {key: next(scalars) for key in fields} for name, fields in zip(names, keys, strict=True)}
    return {"gradient_l2": gradient_norm, **dict(zip(task.components, losses, strict=True)),
            "parameters": parameters, "temperatures": temperatures, "moment_scope": MOMENT_SCOPE}
