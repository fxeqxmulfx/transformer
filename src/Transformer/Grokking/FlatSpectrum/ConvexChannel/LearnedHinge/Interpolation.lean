module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Interpolation.Affine
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Interpolation.Finite
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Interpolation.Spectral

/-!
# Full-word capacity and exact-compression limits of the current learned family

Source: `review/convex/moments.md`, CM84. Actual learned complex neurons give
arbitrary full-word logit tables inside the existing complete convex domain.
The corresponding full response operator has Gram matrix 10I. Preserving all
relative logits with an affine real-coordinate readout requires |O|^T real
dimensions. Preserving every strict classification with any fixed finite-code
forward requires at least |O|^T stored bits. Neither result lower-bounds one
structured task or identifies an unobserved answer. No new learner is proposed.
-/
