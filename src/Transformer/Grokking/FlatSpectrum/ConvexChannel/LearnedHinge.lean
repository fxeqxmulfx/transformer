module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Sequence
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Complete
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Boundary
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.PatternReduction
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Unseen
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Renaming
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Learning
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Replacement
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Interpolation

/-!
# Shared learned complex ReLU representations with explicit convexity scope

Source: `review/flat_spectrum.md`, CM75–CM84. Exact learned-factor recovery,
raw object prefixes, ordinary next-object CE on input activation cones,
complete pattern coverage and its exponential cost, and a counterexample
to extending the convexity theorem outside its declared input region;
exact pattern reduction and actual unseen-answer ambiguity despite flatness.
A shared two-pass restriction provides conditional transfer under renaming.
Finite projection certificates enforce the whole original complex domain.
Complete feasible-query bases retain every admissible direction. Full ordinary
CE pullbacks justify base and shared-pass indexed minibatch update directions.
Local component replacement preserves the whole cone and complete complex forward.
Full-word interpolation gives explicit limits on exact compression of this family.
No native training convergence or stress-test accuracy is claimed.
-/
