module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Projection.Basic

/-!
# Explicit optimal two-coordinate magnitude projection

Source: `review/convex/moments.md`, CM102. All zero and boundary cases occur in
this concrete formula. Optimality is derived from nonnegative excess-distance
expressions, with no optimization hypothesis in the definition.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Projection

/-- CM102: exact magnitude projection for two nonnegative input radii. -/
noncomputable def magnitudes (r : ℝ × ℝ) : ℝ × ℝ :=
  if r.1 + r.2 ≤ 1 then r
  else if 1 ≤ r.1 - r.2 then (1, 0)
  else if 1 ≤ r.2 - r.1 then (0, 1)
  else ((r.1 - r.2 + 1) / 2, (r.2 - r.1 + 1) / 2)

/-- CM102: the formula preserves feasibility and never enlarges a magnitude. -/
theorem magnitude_bounds (r : ℝ × ℝ) (h₁ : 0 ≤ r.1) (h₂ : 0 ≤ r.2) :
    0 ≤ (magnitudes r).1 ∧ (magnitudes r).1 ≤ r.1 ∧
    0 ≤ (magnitudes r).2 ∧ (magnitudes r).2 ≤ r.2 ∧
    (magnitudes r).1 + (magnitudes r).2 ≤ 1 := by
  unfold magnitudes
  split_ifs with h hfirst hsecond
  · exact ⟨h₁, le_rfl, h₂, le_rfl, h⟩
  · dsimp
    constructor
    · norm_num
    constructor
    · linarith
    exact ⟨by norm_num, h₂, by norm_num⟩
  · dsimp
    exact ⟨by norm_num, h₁, by norm_num, by linarith, by norm_num⟩
  · dsimp
    refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> linarith

/-- CM102: the explicit magnitudes minimize the complete squared distance. -/
theorem magnitude_optimal (r : ℝ × ℝ) (h₁ : 0 ≤ r.1) (h₂ : 0 ≤ r.2)
    (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b ≤ 1) :
    (r.1 - (magnitudes r).1) ^ 2 + (r.2 - (magnitudes r).2) ^ 2 ≤
      (r.1 - a) ^ 2 + (r.2 - b) ^ 2 := by
  unfold magnitudes
  split_ifs with h hfirst hsecond
  · nlinarith [sq_nonneg (r.1 - a), sq_nonneg (r.2 - b)]
  · dsimp
    have hp := mul_nonneg (by linarith : 0 ≤ r.1 - 1 - r.2)
      (by linarith : 0 ≤ 1 - a)
    have hq := mul_nonneg h₂ (by linarith : 0 ≤ 1 - a - b)
    nlinarith [sq_nonneg (1 - a), sq_nonneg b]
  · dsimp
    have hp := mul_nonneg (by linarith : 0 ≤ r.2 - 1 - r.1)
      (by linarith : 0 ≤ 1 - b)
    have hq := mul_nonneg h₁ (by linarith : 0 ≤ 1 - a - b)
    nlinarith [sq_nonneg (1 - b), sq_nonneg a]
  · dsimp
    have hp := mul_nonneg (by linarith : 0 ≤ r.1 + r.2 - 1)
      (by linarith : 0 ≤ 1 - a - b)
    nlinarith [sq_nonneg (a - (r.1 - r.2 + 1) / 2),
      sq_nonneg (b - (r.2 - r.1 + 1) / 2)]

example : 0 ≤ (magnitudes (2, 1)).1 ∧ (magnitudes (2, 1)).1 ≤ 2 ∧
    0 ≤ (magnitudes (2, 1)).2 ∧ (magnitudes (2, 1)).2 ≤ 1 ∧
    (magnitudes (2, 1)).1 + (magnitudes (2, 1)).2 ≤ 1 :=
  magnitude_bounds (2, 1) (by norm_num) (by norm_num)

example : (2 - (magnitudes (2, 1)).1) ^ 2 + (1 - (magnitudes (2, 1)).2) ^ 2 ≤
    (2 - (1 / 2 : ℝ)) ^ 2 + (1 - (1 / 2 : ℝ)) ^ 2 :=
  magnitude_optimal (2, 1) (by norm_num) (by norm_num) (1 / 2) (1 / 2)
    (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Projection
