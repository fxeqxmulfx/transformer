"""Exact rational checks for Backend fixture premises, manuscript K6.

Fractions preserve the exact finite binary tensor values. No floating-point
exp, sqrt, sin or cos is used as a mathematical reference. The analytic
implications and remainder bounds are proved in Attention.Backend; this
module checks their rational hypotheses for the finite captured fixtures.
"""

from fractions import Fraction as Q
import math


NAMES = ("rms", "l2", "silu", "sdpa", "cos", "sin", "matrix")
LENGTHS = (4, 4, 1, 4, 2, 2, 2)
EPS = Q(1, 8388608)
TOL = Q(1, 1000000)
RMS_SCALE = Q(36514837, 100000000)
L2_SCALE = Q(18257419, 100000000)


def exact(value):
    """Decode a finite host float; reject nonfinite or nonnumeric records."""
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("a captured value must be a finite number")
    return Q(value)


def exp_polynomial(x, n=12):
    """The exact sum in Backend.expPolynomial, independent of the backend."""
    return sum((x**m / math.factorial(m) for m in range(n)), Q(0))


def rotation_polynomial(x, n=12):
    """Real/imaginary parts of Backend.rotationPolynomial, without complex floats."""
    real = sum(((-1)**(m // 2) * x**m / math.factorial(m)
                for m in range(0, n, 2)), Q(0))
    imag = sum(((-1)**(m // 2) * x**m / math.factorial(m)
                for m in range(1, n, 2)), Q(0))
    return real, imag


def references():
    """Rational witnesses fixed before observing CPU or CUDA outputs."""
    e = exp_polynomial(Q(-1, 2))
    p = 1 / (1 + e)
    phases = (Q(1), Q(1, 100))
    return {
        "rms": [RMS_SCALE * i for i in range(1, 5)],
        "l2": [L2_SCALE * i for i in range(1, 5)],
        "silu": [p / 2],
        "sdpa": [1 - p, p, Q(0), Q(0)],
        "cos": [rotation_polynomial(x)[0] for x in phases],
        "sin": [rotation_polynomial(x)[1] for x in phases],
        "matrix": [Q(5, 4), Q(59, 8)],
    }


def check(outputs):
    """Check every exact rational premise; reject missing sites and bad shapes."""
    if set(outputs) != set(NAMES):
        raise ValueError("captured operation set differs from the fixed fixture")
    if abs(RMS_SCALE**2 * (30 + 4 * EPS) / 4 - 1) > TOL:
        raise ValueError("RMS scale certificate fails")
    if abs(L2_SCALE**2 * 30 - 1) > TOL:
        raise ValueError("L2 scale certificate fails")
    refs = references()
    residuals = {}
    for name, length in zip(NAMES, LENGTHS, strict=True):
        values = outputs[name]
        if len(values) != length:
            raise ValueError(f"{name}: wrong number of coordinates")
        errors = [abs(exact(value) - reference)
                  for value, reference in zip(values, refs[name], strict=True)]
        tolerance = Q(0) if name == "matrix" else TOL
        if max(errors) > tolerance:
            raise ValueError(f"{name}: rational certificate failed")
        residuals[name] = {
            "maximum_exact_residual": str(max(errors)),
            "per_coordinate_allowance": str(tolerance),
            "exact_values": [str(exact(value)) for value in values],
        }
    return residuals
