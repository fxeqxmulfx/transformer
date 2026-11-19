module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection.KernelBasis.Queries

/-!
# Checkable complete representation of feasible query directions

Source: `review/convex/moments.md`, CM81. Finite arithmetic certificates prove
that a proposed basis represents every direction admitted by the CM80 cone
geometry, with unique coefficients and no dropped original coordinates.
Every equality row is retained. This does not prove a native elimination
implementation, finite-difference accuracy, efficient training or convergence.
-/
