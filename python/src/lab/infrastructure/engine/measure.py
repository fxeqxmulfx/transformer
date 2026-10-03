"""Per-tensor measurements around one update; none of them changes the trajectory.

Ports of `paper_reproduction.diagnostics.before_update` and `after_update`.
Moments are read as stored: AdamW's before bias correction, AMSGradW's raw
buffers, which never had one.
"""

import torch

MOMENTS = ("m", "v", "maximum", "exp_avg", "exp_avg_sq")


def moment_scope(optimizer):
    """What the recorded moments are: as a direction optimizer names them, or AdamW's."""
    moments = getattr(optimizer, "moments", None)
    if moments is None and any("exp_avg" in state for state in optimizer.state.values()):
        moments = "AdamW_exp_avg_and_exp_avg_sq_before_bias_correction"
    return "actual_parameter_updates" if moments is None else f"{moments}; actual_parameter_updates"


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
        for key in MOMENTS:
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
    parameters = {name: {key: next(scalars) for key in fields} for name, fields in zip(names, keys, strict=True)}
    return {"gradient_l2": gradient_norm, **dict(zip(task.components, losses, strict=True)),
            "parameters": parameters, "temperatures": temperatures, "moment_scope": moment_scope(optimizer)}
