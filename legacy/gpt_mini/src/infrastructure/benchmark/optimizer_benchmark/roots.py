"""Real-arithmetic DASH solver recipes translated to float tensors.

Scaling, positive regularization, output rescaling and sample-count repair
follow the corrected definitions under src/Transformer/DASH/. Solver budgets
remain finite. This code is independently tested, not extracted from Lean.
"""

import math
import torch


def rayleigh_estimate(matrix, steps=3):
    vector = torch.ones((*matrix.shape[:-1], 1), dtype=matrix.dtype, device=matrix.device)
    for _ in range(steps):
        vector = matrix @ vector
        norm = vector.square().sum(dim=-2, keepdim=True).sqrt()
        vector = vector / torch.where(norm > 0, norm, torch.ones_like(norm))
    numerator = (vector * (matrix @ vector)).sum(dim=(-2, -1))
    denominator = vector.square().sum(dim=(-2, -1))
    return numerator / torch.where(denominator > 0, denominator, torch.ones_like(denominator))


def guarded_scale(matrix, estimate=None):
    frobenius = matrix.square().sum(dim=(-2, -1)).sqrt()
    row_bound = matrix.abs().sum(dim=-1).amax(dim=-1)
    upper = torch.minimum(frobenius, row_bound)
    if estimate is None:
        estimate = rayleigh_estimate(matrix)
    return torch.where(
        (estimate > 0) & (upper < 2 * estimate), 2 * estimate,
        torch.where(upper > 0, 2 * upper, torch.ones_like(upper)),
    )


def ndb_iterate(matrix, steps):
    identity = torch.eye(matrix.size(-1), dtype=matrix.dtype, device=matrix.device)
    root = matrix
    inverse = identity.expand_as(matrix)
    if steps:
        correction = 1.5 * identity - 0.5 * matrix
        root, inverse = matrix @ correction, correction
    for _ in range(1, steps):
        correction = 1.5 * identity - 0.5 * (inverse @ root)
        root, inverse = root @ correction, correction @ inverse
    return root, inverse


def inverse_fourth_ndb(matrix, steps=6):
    scale = guarded_scale(matrix)
    normalized = matrix / scale[..., None, None]
    first_root, _ = ndb_iterate(normalized, steps)
    _, inverse = ndb_iterate(first_root, steps)
    return inverse * scale.pow(-0.25)[..., None, None]


def inverse_fourth_cn(matrix, steps=8):
    identity = torch.eye(matrix.size(-1), dtype=matrix.dtype, device=matrix.device)
    scale = guarded_scale(matrix)
    normalized = matrix / scale[..., None, None]
    unit_scale = 5 ** (-0.25)
    x = identity.expand_as(matrix) / unit_scale
    residual = normalized / unit_scale ** 4
    for _ in range(steps):
        correction = 1.25 * identity - 0.25 * residual
        x = x @ correction
        squared = correction @ correction
        residual = (squared @ squared) @ residual
    return x * scale.pow(-0.25)[..., None, None]


def inverse_fourth_evd(matrix):
    values, vectors = torch.linalg.eigh(matrix)
    # Histories are regularized SPD; this is only a roundoff protection.
    values = values.clamp_min(torch.finfo(matrix.dtype).tiny)
    return (vectors * values.pow(-0.25).unsqueeze(-2)) @ vectors.transpose(-2, -1)


_cosine_cache = {}


def fitted_power(matrix, epsilon, exponent=-0.25, degree=60, samples=1000):
    """Actual cosine fit and optimized Clenshaw for (A + epsilon I)^exponent."""
    samples = max(samples, degree + 1)
    scale = guarded_scale(matrix)
    identity = torch.eye(matrix.size(-1), dtype=matrix.dtype, device=matrix.device)
    argument = 2 * matrix / scale[..., None, None] - identity
    key = (str(matrix.device), matrix.dtype, degree, samples)
    if key not in _cosine_cache:
        angles = (2 * torch.arange(samples, dtype=matrix.dtype, device=matrix.device) + 1)
        angles = angles * (math.pi / (2 * samples))
        modes = torch.arange(degree + 1, dtype=matrix.dtype, device=matrix.device)
        _cosine_cache[key] = angles.cos(), (angles[:, None] * modes).cos()
    nodes, basis = _cosine_cache[key]
    values = (epsilon / scale[..., None] + (nodes + 1) / 2).pow(exponent)
    coefficients = (2 / samples) * (values @ basis)
    coefficients[..., 0] *= 0.5
    if degree == 0:
        output = coefficients[..., 0, None, None] * identity
    elif degree == 1:
        output = coefficients[..., 1, None, None] * argument
        output = output + coefficients[..., 0, None, None] * identity
    else:
        second = coefficients[..., degree, None, None] * identity
        first = 2 * coefficients[..., degree, None, None] * argument
        first = first + coefficients[..., degree - 1, None, None] * identity
        for k in range(degree - 2, 0, -1):
            first, second = (2 * (argument @ first) - second
                             + coefficients[..., k, None, None] * identity), first
        output = argument @ first - second + coefficients[..., 0, None, None] * identity
    return output * scale.pow(exponent)[..., None, None]
