"""Read raw-prefix answer features with temporary hooks.

Source: Nanda et al., arXiv:2301.05217v1, sections 4 and 5.1. Deviation:
capture division features at the equals token, including actual FFN
post-activation neurons, without choosing final-model Fourier frequencies.
"""

from contextlib import contextmanager

import torch


@contextmanager
def evaluating(model):
    """Restore mixed module modes and all framework RNG after observation."""
    modes = {module: module.training for module in model.modules()}
    device = next(model.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        for module, mode in modes.items():
            module.training = mode


class Capture:
    def __init__(self, model):
        self.model, self.chunks, self.handles = model, {}, []

    def keep(self, name, values):
        self.chunks.setdefault(name, []).append(values[:, 4].detach().cpu())

    def __enter__(self):
        def output(name):
            return lambda module, inputs, values: self.keep(name, values)

        def neurons(name):
            return lambda module, inputs: self.keep(name, inputs[0])

        for name, module in self.model.named_modules():
            parts = name.split(".")
            if name in ("embed", "final_norm"):
                self.handles.append(module.register_forward_hook(output(name)))
            if len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit():
                if len(parts) == 2 or (len(parts) == 3 and parts[2] in ("attention", "ffn")):
                    self.handles.append(module.register_forward_hook(output(name)))
                if len(parts) == 4 and parts[2:] == ["ffn", "output"]:
                    self.handles.append(module.register_forward_pre_hook(neurons(f"blocks.{parts[1]}.neurons")))
        return self

    def __exit__(self, *error):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def result(self):
        return {name: torch.cat(chunks) for name, chunks in self.chunks.items()}


@torch.no_grad()
def forward(model, inputs, batch=512):
    device = next(model.parameters()).device
    return torch.cat([model(chunk.to(device))[:, 4].detach().cpu() for chunk in inputs.split(batch)])


def collect(model, inputs, batch=512):
    with evaluating(model), Capture(model) as captured:
        logits = forward(model, inputs, batch)
    return logits, captured.result()
