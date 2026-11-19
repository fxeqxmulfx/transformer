module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection.Joint
public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection.Step
public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection.Mixture
public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection.KernelBasis

/-!
# Exact certificates for whole-model feasible learning updates

Source: `review/convex/moments.md`, CM79–CM81. Finite arithmetic certificates prove
nearest-point projection onto the original full complex activation domain.
A projected-step work inequality and an explicitly conditional descent bound
support future native Mini and forward-only Mini updates. Projected normals
identify every forced equality and certify two-sided forward-query directions,
including degenerate cones. Positive centre mixtures have explicit displacement
and convex-loss bounds. Finite basis certificates prove complete coverage of
the admissible query directions. The construction
retains every model coordinate and does not assert an efficient solver,
an optimizer trajectory or any new stress-test accuracy.
-/
