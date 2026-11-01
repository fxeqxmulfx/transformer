"""PyTorch modules for model specs."""

import math

import torch

from ...domain.gradient_paths import BranchGates, LinearTransport
from .gradient_paths import build as build_gradient_model
from ...domain import model
from ...domain.atomic import AtomicMatching as AtomicSpec
from ...domain.regression import MLP as MLPSpec
from ...domain.paired import VectorReadout as VectorReadoutSpec
from ...domain.rotary import Rotate90 as Rotate90Spec
from ...domain.peri_rotary import PeriRotate90 as PeriRotate90Spec
from ...domain.peri_restart import PeriRotate90Restart
from ...domain.source_transformer import SourceTransformer as SourceSpec
from ...domain.tensor import TensorStack as TensorSpec
from .atomic import AtomicMatching
from .mlp import MLP
from .rotary import Rotate90
from .peri_rotary import PeriRotate90
from .peri_restart import restart
from .tensor import TensorStack
from .transformer import Transformer
from .source_transformer import SourceTransformer
from ...domain.residual_transformer import ResidualTransformer as ResidualSpec
from .residual_transformer import ResidualTransformer


def build_model(spec, vocab, seed):
    """Seed the global generator, construct, initialize; float32 on the CPU.

    The order matches `paper_reproduction.grokking.make_model`, and the
    initialization of the convex MQAR `RopeTransformer`, so a model of the
    same spec and seed starts from bit-identical parameters.
    """
    torch.manual_seed(seed)
    if isinstance(spec, (BranchGates, LinearTransport)):
        return build_gradient_model(spec)
    if isinstance(spec, ResidualSpec):
        return ResidualTransformer(spec, vocab)
    if isinstance(spec, SourceSpec):
        return SourceTransformer(spec, vocab)
    if isinstance(spec, PeriRotate90Restart):
        return restart(spec, vocab, seed)
    if isinstance(spec, PeriRotate90Spec):
        return PeriRotate90(spec, vocab, seed)
    if isinstance(spec, Rotate90Spec):
        return Rotate90(spec, vocab, seed)
    if isinstance(spec, AtomicSpec):
        return AtomicMatching(spec, vocab)
    if isinstance(spec, TensorSpec):
        return TensorStack(spec, vocab)
    vector = isinstance(spec, VectorReadoutSpec)
    spec = spec.network if vector else spec
    built = MLP(spec, vocab, outputs=vocab if vector else 1) if isinstance(spec, MLPSpec) else Transformer(spec, vocab)
    if not isinstance(spec.init, (model.TorchDefault, model.Normal, model.ScaledResidual)):
        raise NotImplementedError(f"No builder for {spec.init!r}")
    if isinstance(spec.init, (model.Normal, model.ScaledResidual)):
        for parameter in built.parameters():
            if parameter.ndim >= 2:
                torch.nn.init.normal_(parameter, std=spec.init.std)
    if isinstance(spec.init, model.ScaledResidual):
        for module in built.modules():
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        for block in built.blocks:
            torch.nn.init.normal_(block.attention.output.weight, std=spec.init.std / math.sqrt(2 * spec.depth))
            torch.nn.init.normal_(block.ffn.output.weight, std=spec.init.std / math.sqrt(2 * spec.depth))
    return built.to(torch.float32)
