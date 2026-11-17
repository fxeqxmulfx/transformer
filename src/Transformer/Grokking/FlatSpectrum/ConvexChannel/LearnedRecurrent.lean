module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Domain
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Queries
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Scalar
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Parity
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Pair

/-!
# Learned complex recurrence, bounded convex mixtures and next-object prediction

Source: `review/flat_spectrum.md`, CM95–97. Complete learned transition execution,
depth-independent operator bounds, joint-domain CE convexity, feasible finite
loss queries and a nonconstant arbitrary-length parity witness. This removes
the fixed-interaction expressivity obstruction without claiming optimizer
discovery or generalization from arbitrary incomplete observations. The scalar
subclass loses order; a complete two-state matrix family retains it.
-/
