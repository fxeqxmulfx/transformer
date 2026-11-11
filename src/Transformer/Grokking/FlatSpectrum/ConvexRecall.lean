module

public import Transformer.Grokking.FlatSpectrum.ConvexRecall.Defs
public import Transformer.Grokking.FlatSpectrum.ConvexRecall.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexRecall.Loss
public import Transformer.Grokking.FlatSpectrum.ConvexRecall.Training
public import Transformer.Grokking.FlatSpectrum.ConvexRecall.Separation
public import Transformer.Grokking.FlatSpectrum.ConvexRecall.MQAR
public import Transformer.Grokking.FlatSpectrum.ConvexRecall.Spectrum
public import Transformer.Grokking.FlatSpectrum.ConvexRecall.Convergence

/-!
# Convex answer-loss recall

Source: `review/flat_spectrum.md`, *Convex answer-loss recall*, CR1-CR7.
A directly learned stochastic key-matching matrix with fixed categorical
value copying has a convex feasible set and jointly linear logits. Its
ordinary answer cross-entropy is convex. Separating training dictionaries
make identity matching its unique global optimizer; that optimizer solves
every dictionary and every aligned distinct-key causal MQAR stream.
The output kernel is flat for centered scalar readouts, while remaining
incompatible with nuclear-norm completion from a proper observation set.
Feasible loss-minimizing sequences converge to the identity and eventually
recall every dictionary by strict argmax. A particular optimization
algorithm's loss convergence is not assumed to follow from these facts.
The construction retains full key capacity and fixed token identities;
it is not a convexification of arbitrary learned embeddings or transformers.
-/
