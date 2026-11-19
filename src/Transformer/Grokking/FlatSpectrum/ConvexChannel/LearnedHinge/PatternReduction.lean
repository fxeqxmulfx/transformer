module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.PatternReduction.Cost

/-!
# Exact reduction of the complete complex activation-pattern family

Source: `review/convex/moments.md`, CM76. Aggregate real and imaginary
components independently, retain the full complex forward, embed the result
back into the original family, and specialize to every raw bounded prefix.
All scalar patterns remain; the remaining exponential cost is explicit.
No optimizer, native peak-memory claim or unseen-answer learning is asserted.
-/
