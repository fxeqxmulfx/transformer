module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Defs
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Learning
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Training
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Work
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Boolean
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Minibatch
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Inference

/-!
# A finite forward-only learner for realizable generated channels

Source: `review/flat_spectrum.md`, *Forward-only channel learning*, CF1-CF6.
The executable categorical learner eliminates keys with contradictory
answers. Realizability guarantees a successful return and a global ordinary
CE optimum. Collisions at wrong keys and coverage at the true key imply
recovery of the entire generated rule, including held-out dictionaries.
An instrumented implementation proves the number of candidate observations
is bounded by the product of the candidate and sample counts. No gradient,
optimizer convergence assumption, or unproved spectral generalization is used.
Persistent minibatches compute exactly the same result as one uninterrupted scan.
Inference-time feedback and the limits of self-labeling are formalized separately
under *Inference-time learning*, CI1-CI4.
-/
