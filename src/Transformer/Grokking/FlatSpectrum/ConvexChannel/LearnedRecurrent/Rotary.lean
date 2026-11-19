module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Execution
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Domain
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Queries
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Order
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Fixed
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Minibatch
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout

/-!
# Learned rotary positions in the bounded next-object recurrence

Source: `review/convex/moments.md`, CM98–100. Actual position-dependent matrix
execution, unchanged operator capacities, a joint convex score domain, feasible
finite CE queries and an order-sensitive witness. No unseen-label or optimal
component-discovery guarantee is inferred from norm preservation.
Fixed-frequency controls retain feasible CE queries. A concrete minibatch
descent step can worsen combined CE; capped steps preserve capacity only.
Complete readout refitting on already learned states is a conditional convex
subproblem, with actual feasible radial updates and full-family CE queries.
-/
