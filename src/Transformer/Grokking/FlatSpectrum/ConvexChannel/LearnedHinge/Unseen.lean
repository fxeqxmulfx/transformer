module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Unseen.Training
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Unseen.Witness
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Unseen.Spectral

/-!
# Unseen-answer ambiguity in the actual learned complex sequence model

Source: `review/convex/moments.md`, CM77. A learned raw-prefix neuron gives
an exact feasible perturbation of any fitted point at one omitted full prefix.
Every observed logit and indexed ordinary CE stays equal. Nonconstant fitted
examples and an actual flat observed operator with a blind direction show
why representation coverage and flatness do not identify the missing answer.
-/
