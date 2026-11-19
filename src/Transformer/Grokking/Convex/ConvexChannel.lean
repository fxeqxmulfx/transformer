module

public import Transformer.Grokking.Convex.ConvexChannel.Defs
public import Transformer.Grokking.Convex.ConvexChannel.Basic
public import Transformer.Grokking.Convex.ConvexChannel.Representation
public import Transformer.Grokking.Convex.ConvexChannel.Optimality
public import Transformer.Grokking.Convex.ConvexChannel.Identification
public import Transformer.Grokking.Convex.ConvexChannel.Capacity
public import Transformer.Grokking.Convex.ConvexChannel.Limits
public import Transformer.Grokking.Convex.ConvexChannel.Oracle
public import Transformer.Grokking.Convex.ConvexChannel.Forward
public import Transformer.Grokking.Convex.ConvexChannel.Observation
public import Transformer.Grokking.Convex.ConvexChannel.Batch
public import Transformer.Grokking.Convex.ConvexChannel.Minibatch
public import Transformer.Grokking.Convex.ConvexChannel.Shared
public import Transformer.Grokking.Convex.ConvexChannel.SharedLearning
public import Transformer.Grokking.Convex.ConvexChannel.HiddenPath
public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge
public import Transformer.Grokking.Convex.ConvexChannel.LearnedQuadratic
public import Transformer.Grokking.Convex.ConvexChannel.LearnedBilinear
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent
public import Transformer.Grokking.Convex.ConvexChannel.LearnedRetrieval
public import Transformer.Grokking.Convex.ConvexChannel.ComposedRetrieval
public import Transformer.Grokking.Convex.ConvexChannel.Adaptive
public import Transformer.Grokking.Convex.ConvexChannel.Completion
public import Transformer.Grokking.Convex.ConvexChannel.Counts
public import Transformer.Grokking.Convex.ConvexChannel.CausalPairs

/-!
# Convex channels with flat-spectrum capacity bounds

Source: `review/convex/channels.md`, *Convex channels and capacity*, CC1-CC8.
Directly learned joint routing/output weights extend fixed copying while
keeping the feasible set convex, the forward map linear in all weights,
and ordinary answer cross-entropy convex. Separating answer examples
identify generated recall. Flat spectrum bounds the actual categorical
feature width and approximation error; it does not identify unseen labels.
An exact linear minimization oracle supports convex feasible updates.
Observed-feature reconstruction certificates identify unseen predictions
and bound error amplification; see the companion derivation O1-O4.
CM76-CM95 add actual learned complex encoders and recurrence while retaining
every original output object. Their convexity statements concern joint tensor
or function domains, not independently varied physical factor arrays. No
arithmetic features or claim that an arbitrary flat task generalizes from an
arbitrary sample are introduced.
-/
