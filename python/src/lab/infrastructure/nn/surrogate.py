"""Declared surrogate score gradients for EXPERIMENT_PLAN.md, step 3.

Both normalizers use the same scores. Their first-order Jacobians are
those of arXiv:1602.02068v2, section 2.4. The difference between a tensor
and its detached copy is exactly zero for these finite probabilities,
while autograd follows only its attached copy. Thus routing and value
gradients follow the forward normalizer; score gradients follow the other.
This mixed rule is an intervention, not that forward's true derivative.
Backward probabilities are cast to the forward's dtype before the zero
difference; the normalizers retain their existing precision policies.
"""


def surrogate_weights(scores, forward, backward):
    """Preserve forward weights exactly and differentiate the other normalizer."""
    weights = forward(scores)
    sensitivity = backward(scores).to(weights.dtype)
    return weights.detach() + (sensitivity - sensitivity.detach())
