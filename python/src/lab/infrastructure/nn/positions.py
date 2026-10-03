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
    """`experiments/gpt_mini.py` `rope_tables`: cos and sin of shape (context, head_width / 2)."""
    half = head_width // 2
    inverse = theta ** (-torch.arange(half, dtype=torch.float32) / half)
    angles = torch.outer(torch.arange(context, dtype=torch.float32), inverse)
    return angles.cos(), angles.sin()


def rotate(x, cos, sin):
    """Rotate the two halves of the last axis; cos and sin broadcast over leading axes."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)
