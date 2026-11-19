module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Renaming.Prediction
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Renaming.Witness
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Renaming.Coverage

/-!
# A shared learned complex sequence model with conditional unseen transfer

Source: `review/convex/moments.md`, CM78. Two actual learned passes share all
parameters and align outputs under an involutive object renaming. Ordinary
answer CE remains convex on the full input domain. Low observed CE identifies
the next original object at the renamed prefix under an explicit task symmetry.
A nonconstant witness is inhabited, and fixed inputs expose the restriction.
The complete parameter family remains exponential; no native update or new
stress-test accuracy is asserted.
-/
