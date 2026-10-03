"""Position tables. Each is computed exactly as its historical source computed it."""

import numpy as np
import torch


def sinusoid_table(context, width, base):
    """openai/grok `Transformer._position_encoding`: sin at even channels, cos at odd.

    NumPy evaluates each entry in float64; the table is then rounded to float32.
    """
    rows = [torch.tensor([np.sin(position / (base ** (channel / width))) if channel % 2 == 0
                          else np.cos(position / (base ** ((channel - 1) / width)))
                          for channel in range(width)])
            for position in range(context)]
    return torch.stack(rows).to(torch.float32)


def rope_tables(head_width, context, theta):
    """`experiments/gpt_mini.py` `rope_tables`: cos and sin of shape (context, head_width / 2).

    The frequencies equal those of the convex MQAR `RotaryAttention`, bit for bit.
    """
    half = head_width // 2
    inverse = theta ** (-torch.arange(half, dtype=torch.float32) / half)
    angles = torch.outer(torch.arange(context, dtype=torch.float32), inverse)
    return angles.cos(), angles.sin()


def rotate(x, cos, sin, interleaved):
    """Rotate pairs of the last axis; cos and sin broadcast over leading axes.

    The pairs are the two halves (GPTMini), or interleaved even and odd
    channels (the convex MQAR `RotaryAttention.rotate`).
    """
    if interleaved:
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
    first, second = x.chunk(2, dim=-1)
    return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)
