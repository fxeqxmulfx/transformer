"""RMSNorm with the same forward and a separated radial/tangent backward.

Source: Transformer.Residual.radialRMS_hasDerivAt and task155 benchmark
Eqs. (1)–(3). For energy m>0 and p=x*mean(x*g)/m, the VJP is
inv*(g-p) + eps*inv**3*p. Computing the projection in FP64 avoids erasing
the small positive radial term by subtracting two large terms in FP32.
The returned gradient has the input dtype. Later upstream reductions can
still round, and a mathematically tiny gradient can remain poorly conditioned.
"""

import torch
from torch.nn import functional as F


class _RMSBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, width, eps):
        ctx.save_for_backward(value)
        ctx.eps = eps
        return F.rms_norm(value, (width,), eps=eps)

    @staticmethod
    def backward(ctx, upstream):
        value, = ctx.saved_tensors
        value64, upstream64 = value.double(), upstream.double()
        energy = value64.square().mean(-1, keepdim=True)
        denominator = torch.where(energy > 0, energy, torch.ones_like(energy))
        parallel = value64 * (value64 * upstream64).mean(-1, keepdim=True) / denominator
        inverse = (energy + ctx.eps).rsqrt()
        gradient = inverse * (upstream64 - parallel) + ctx.eps * inverse.pow(3) * parallel
        return gradient.to(value.dtype), None, None


def rms_norm(value, width, eps):
    """Explicit positive epsilon, parameter-free final-axis RMS normalization."""
    return _RMSBackward.apply(value, width, eps)
