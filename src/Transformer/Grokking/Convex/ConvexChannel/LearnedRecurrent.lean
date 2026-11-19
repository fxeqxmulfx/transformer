module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Basic
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Domain
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Queries
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Scalar
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Parity
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Pair
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Hilbert
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Width

/-!
# Learned complex recurrence, bounded convex mixtures and next-object prediction

Source: `review/convex/moments.md`, CM95–98 and CM107. Complete learned transition execution,
depth-independent operator bounds, joint-domain CE convexity, feasible finite
loss queries and a nonconstant arbitrary-length parity witness. This removes
the fixed-interaction expressivity obstruction without claiming optimizer
discovery or generalization from arbitrary incomplete observations. The scalar
subclass loses order; a complete two-state matrix family retains it.
Learned rotary positions preserve actual operator bounds and can distinguish
orders even with commuting content matrices.
CM107 adds a complete Hilbert-state branch without dropping old rotary points;
the old row geometry can contract a norm-preserving mixing operator.
CM110 separates larger state width from arbitrarily many plain two-state blocks.
-/
