"""PyTorch modules for model specs."""

import torch

from ...domain import model
from .transformer import Transformer


def build_model(spec, vocab, seed):
    """Seed the global generator, construct, initialize; float32 on the CPU.

    The order matches `paper_reproduction.grokking.make_model`, so a model of
    the same spec and seed starts from bit-identical parameters.
    """
    torch.manual_seed(seed)
    built = Transformer(spec, vocab)
    if isinstance(spec.init, model.Normal):
        for parameter in built.parameters():
            if parameter.ndim >= 2:
                torch.nn.init.normal_(parameter, std=spec.init.std)
    elif not isinstance(spec.init, model.TorchDefault):
        raise NotImplementedError(f"No builder for {spec.init!r}")
    return built.to(torch.float32)
