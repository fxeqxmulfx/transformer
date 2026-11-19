module

public import Transformer.Grokking.Convex.CoupledMoments.DensityExtraction.Examples

/-!
# Convex matrix extraction from an exact Fourier subspace

Source: `review/convex/moments.md`, CM21. PSD trace-one matrices with affine
constant-diagonal readings form exactly the hull of pure mode projectors when
frequency differences are separated. Two observable linear objectives select
a pure mode in a supplied rotated basis; centering and a prime modulus give
distinct phases at every row. Exact subspace recovery from incomplete labels,
numerical convergence, and unseen decoded labels remain separate questions.
-/

@[expose] public section
