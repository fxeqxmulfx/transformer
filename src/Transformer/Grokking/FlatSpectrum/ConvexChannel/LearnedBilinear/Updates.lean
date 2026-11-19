module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Updates.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Updates.Forward
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Updates.Support
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Updates.Scaling
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Updates.Unobserved

/-!
# Complete finite-query minibatch updates and exact zero storage

Source: `review/convex/moments.md`, CM88. Actual learned logit identities
permit exact reuse of forward CE queries across all complete coordinates.
Zero support is represented implicitly only after proving its update and
recovery identities. Finite differences and numerical spectral recovery
remain approximations; no optimizer convergence is asserted.
-/
