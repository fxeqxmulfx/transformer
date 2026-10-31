module

public import Transformer.Rotate90.Forcing.Actual

/-!
# Actual gradient forcing beyond decay and averaging

Source: `review/flat_spectrum.md`, *Gradient forcing*, F1--F4. The actual
two-buffer source is an ordered scalar matrix evolution plus transported
normalized-gradient impulses. Zero-gradient decay retains its raw moments
and stays in the initial query/base span. The full finite network's gradients
provide the forcing, and its nonlinear forward evaluates the resulting weights.
Escaping a parameter subspace requires an input direction outside it; this
is a necessary geometric condition, not a generalization guarantee.
-/
