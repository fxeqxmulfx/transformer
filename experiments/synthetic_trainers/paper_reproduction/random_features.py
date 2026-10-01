"""Complex random Fourier features and the zero-initialized gradient-flow limit.

Source: Deep Double Descent, arXiv:1912.02292v1, Appendix C, Figures 14–15.
The paper specifies exp(-i*x), a frozen Gaussian first layer with variance 1/d,
and a zero-initialized MSE head trained by gradient flow. Here QR computes its
minimum-norm least-squares limit; this does not reproduce finite-time curves.
"""

import torch
import torch.nn.functional as F


def features(images, gaussian, width):
    if width < 1 or width > gaussian.shape[1] or images.shape[1] != gaussian.shape[0]:
        raise ValueError("Invalid random-feature shape or width")
    phase = images @ (gaussian[:, :width] / width ** .5)
    return torch.polar(torch.ones_like(phase), -phase)


def fit_head(design, labels, classes=10):
    """Minimum-norm complex least squares for a numerically full-rank design.

    Overdetermined QR projects labels onto the feature span. Underdetermined
    QR of Z* keeps the head in range(Z*), the same subspace as gradient flow
    from zero. No ridge, clipping, or test-driven regularization is added.
    Pivot and residual diagnostics remain visible instead of silently hiding
    a numerical interpolation failure.
    """
    if design.ndim != 2 or design.shape[0] != len(labels) or min(design.shape) < 1:
        raise ValueError("Design and label rows must match")
    targets = F.one_hot(labels.long(), classes).to(design.dtype)
    rows, columns = design.shape
    if rows >= columns:
        orthogonal, triangular = torch.linalg.qr(design, mode="reduced")
        head = torch.linalg.solve_triangular(triangular, orthogonal.mH @ targets, upper=True)
    else:
        orthogonal, triangular = torch.linalg.qr(design.mH, mode="reduced")
        head = orthogonal @ torch.linalg.solve_triangular(triangular.mH, targets, upper=False)
    if not torch.isfinite(head).all():
        raise FloatingPointError("Nonfinite least-squares head")
    residual = design @ head - targets
    return head, {"minimum_QR_pivot": triangular.diagonal().abs().min().item(),
                  "maximum_target_residual": residual.abs().max().item(),
                  "normal_equation_residual": (design.mH @ residual).abs().max().item(),
                  "head_norm": torch.linalg.vector_norm(head).item(),
                  "interpolation_MSE": residual.abs().square().mean().item(),
                  "solver": "complex_QR_minimum_norm_gradient_flow_limit"}


def score(design, head, labels):
    prediction = design @ head
    target = F.one_hot(labels.long(), head.shape[1]).to(design.dtype)
    return {"error": (prediction.real.argmax(dim=1) != labels).double().mean().item(),
            "MSE": (prediction - target).abs().square().mean().item(),
            "imaginary_prediction_MSE": prediction.imag.square().mean().item(),
            "examples": len(labels)}


def finite_gradient_flow(design, labels, time, classes=10):
    """Spectral solution of dW/dt = -Z*(ZW-Y)/N from W(0)=0."""
    if time < 0:
        raise ValueError("Gradient-flow time must be nonnegative")
    left, singular, right = torch.linalg.svd(design, full_matrices=False)
    target = F.one_hot(labels.long(), classes).to(design.dtype)
    denominator = singular.clamp_min(torch.finfo(singular.dtype).tiny)
    factors = -torch.expm1(-time * singular.square() / design.shape[0]) / denominator
    return right.mH @ (factors[:, None] * (left.mH @ target))
