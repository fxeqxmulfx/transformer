module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.MarginGuard.Capacity

/-!
# Convex constraints preserving fixed raw decisions

Source: `review/flat_spectrum.md`, CM47. Intersect the actual affine CM46
model with fixed raw-margin halfspaces. A maximizing reference is feasible,
and ordinary full-vocabulary CE remains convex. Positive retention preserves
the least maximizing raw label, including ties. Nonzero Fourier examples
change confidence while preserving a wrong reference label, and show why
zero retention does not preserve the deterministic decision.
-/
