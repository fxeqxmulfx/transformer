module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Learning.Examples

/-!
# Actual complex CE directions for complete minibatch learning

Source: `review/convex/moments.md`, CM82. Complete coordinate pullbacks support
the actual next-object CE on the original activation domain and equal its
derivative along certified two-sided queries. Both shared passes, every output
class, repeated records and short minibatches are retained. Nonzero witnesses
exercise every conditional theorem. Finite-step estimation, native optimizer
convergence and full-family resource admission are not established here.
-/
