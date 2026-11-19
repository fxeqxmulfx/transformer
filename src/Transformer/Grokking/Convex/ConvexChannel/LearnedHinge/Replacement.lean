module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Replacement.Witness

/-!
# Certified local replacement inside the complete learned complex model

Source: `review/convex/moments.md`, CM83. Rechecking every original row of
one changed component suffices when the entire old model is feasible.
Actual inference retains every neuron and substitutes the complete new
complex neuron in its original summation position. Nonzero examples cover
real and imaginary components and both readout signs. Native binary64
equivalence and training speed remain separate implementation obligations.
-/
