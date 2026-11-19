module

public import Transformer.Grokking.Convex.ConvexChannel.Minibatch.Gradient
public import Transformer.Grokking.Convex.ConvexChannel.Minibatch.Failure
public import Transformer.Grokking.Convex.ConvexChannel.Minibatch.Memory

/-!
# The singleton minibatch failure and a gradient-memory repair

Source: `review/convex/channels.md`, *Repairing minibatch and full-batch failures*,
MF1–MF2. Exact singleton CE oracles can remain at fifty percent on representable
data. Retaining both gradients removes the demonstrated bad oracle tie even
when those gradients were evaluated at different iterates. This establishes a
specific repair mechanism, not general stochastic convergence.
-/
