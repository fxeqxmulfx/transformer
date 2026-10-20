"""Context extension that preserves the source model's rotary metadata."""

from dataclasses import replace

from .model import RoPE, Transformer, YaRN
from .spec import require


def extend_context(model, context):
    """Enable derived YaRN while copying the original RoPE configuration.

    Source: YaRN's Section 3.4 fixed-context rule (arXiv:2309.00071v3),
    Transformer.RoPE.YaRN.contextScale at 97e19a48 and retentionGain at
    4f4bbc61. Temperature is the derived variant documented by YaRN.
    A second extension keeps the original training context, not the
    intermediate target length. This creates a spec; existing parameter
    tensors can be loaded into the built model without renaming them.
    """
    if not isinstance(model, Transformer) or not isinstance(model.positions, (RoPE, YaRN)):
        raise TypeError("Context extension needs a Transformer with RoPE or YaRN positions")
    require(type(context) is int and context >= model.context,
            "Extended context must be an integer at least as large as the current context")
    original = model.positions.original_context if isinstance(model.positions, YaRN) else model.context
    positions = YaRN(original_context=original, theta=model.positions.theta, interleaved=model.positions.interleaved)
    return replace(model, context=context, positions=positions)
