module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedQuadratic.Basic
public import Transformer.Grokking.Convex.ConvexChannel.LearnedQuadratic.Model
public import Transformer.Grokking.Convex.ConvexChannel.LearnedQuadratic.Sequence
public import Transformer.Grokking.Convex.ConvexChannel.LearnedQuadratic.Witness
public import Transformer.Grokking.Convex.ConvexChannel.LearnedQuadratic.Restriction
public import Transformer.Grokking.Convex.ConvexChannel.LearnedQuadratic.Spectral

/-!
# A polynomial-size learned quadratic sequence family with explicit limits

Source: `review/convex/moments.md`, CM85. Full complex PSD coordinates recover
actual learned representations, squared-magnitude activations, and readouts.
Ordinary next-original-object CE and all indexed minibatches are jointly
convex. The immutable sequence interface retains all positions and objects.
Physical factors of arbitrary finite width are covered with an explicit
full-dimensional recovery; no small fixed-rank bottleneck is preserved.
The complete dense parameter count is polynomial instead of exponential.
Nonzero complex witnesses change both layers and relative predictions.
Three-bit parity proves that the smaller family loses higher interactions;
R3 supplies only a necessary flat-spectrum bound on its actual complex output.
No native learning convergence, unseen-answer identification, or stress-test
accuracy is claimed for this new family.
-/
