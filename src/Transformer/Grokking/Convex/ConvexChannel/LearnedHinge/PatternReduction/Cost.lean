module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.PatternReduction.Witness

/-!
# Exact parameter-array costs before and after pattern reduction

Source: `review/convex/moments.md`, CM76. These counts concern all binary64
real parameter coordinates, not optimizer buffers, constraints or a native
run's peak memory. Fifteen region inputs, two classes and seven coordinates
give a concrete change in admission; exponential growth still remains.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.PatternReduction

/-- Complete old parameter-array bytes for one old hidden slot, fifteen
region inputs, two classes and seven complex coordinates, CM76. -/
theorem old_fifteen_input_bytes :
    (8 * 4 * 1 * 2 * 7 * 2 ^ (2 * 15) : ℕ) = 481036337152 := by
  norm_num

/-- The exactly equivalent reduced parameter array has this many bytes,
CM76. Both complex components, classes and signs are charged. -/
theorem reduced_fifteen_input_bytes :
    (8 * 4 * 2 * 7 * 2 ^ 15 : ℕ) = 14680064 := by
  norm_num

/-- This complete parameter array crosses the user's RAM boundary after
exact reduction, CM76. This does not admit uncounted buffers or computation. -/
theorem fifteen_input_array_boundary :
    14680064 < (110000000000 : ℕ) ∧ 110000000000 < 481036337152 := by
  norm_num

/-- Even the reduced family remains exponential: one binary64 complex
coordinate per scalar pattern at thirty-three inputs exceeds the cap, CM76. -/
theorem thirty_three_inputs_exceed_ram : (110000000000 : ℕ) < 16 * 2 ^ 33 := by
  norm_num

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.PatternReduction
