module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Execution
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Domain
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Queries
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Order

/-!
# Learned rotary positions in the bounded next-object recurrence

Source: `review/flat_spectrum.md`, CM98. Actual position-dependent matrix
execution, unchanged operator capacities, a joint convex score domain, feasible
finite CE queries and an order-sensitive witness. No unseen-label or optimal
component-discovery guarantee is inferred from norm preservation.
-/
