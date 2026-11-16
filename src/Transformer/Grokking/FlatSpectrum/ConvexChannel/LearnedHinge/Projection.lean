module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection.Joint
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection.Step

/-!
# Exact certificates for whole-model feasible learning updates

Source: `review/flat_spectrum.md`, CM79. Finite arithmetic certificates prove
nearest-point projection onto the original full complex activation domain.
A projected-step work inequality and an explicitly conditional descent bound
support future native Mini and forward-only Mini updates. The construction
retains every model coordinate and does not assert an efficient solver,
an optimizer trajectory or any new stress-test accuracy.
-/
