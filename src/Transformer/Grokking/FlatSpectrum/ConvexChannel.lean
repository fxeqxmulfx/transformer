module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Defs
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Representation
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Optimality
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Identification
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Capacity
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Limits
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Oracle
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Observation
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Batch
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Minibatch
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Shared
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Adaptive
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Completion
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Counts

/-!
# Convex channels with flat-spectrum capacity bounds

Source: `review/flat_spectrum.md`, *Convex channels and capacity*, CC1-CC8.
Directly learned joint routing/output weights extend fixed copying while
keeping the feasible set convex, the forward map linear in all weights,
and ordinary answer cross-entropy convex. Separating answer examples
identify generated recall. Flat spectrum bounds the actual categorical
feature width and approximation error; it does not identify unseen labels.
An exact linear minimization oracle supports convex feasible updates.
Observed-feature reconstruction certificates identify unseen predictions
and bound error amplification; see the companion derivation O1-O4.
No arithmetic features, factorized learned embeddings, or claim that an
arbitrary flat task generalizes from an arbitrary sample are introduced.
-/
