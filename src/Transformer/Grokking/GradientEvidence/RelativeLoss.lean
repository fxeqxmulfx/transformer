module

public import Transformer.Grokking.GradientEvidence.JointMargin

/-!
# Cross-entropy change is a probability-weighted exponential moment

Source: `review/flat_spectrum.md`, *Joint difficult-query control*, JC3.
Old softmax probabilities weight each relative logit change. This exact
identity preserves signed contributions and does not replace individual
query losses by an average classification margin.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.GradientEvidence

open Finset Geometry Landscape.Neural

variable {C : Type*} [Fintype C]

/-- Exponential moment weighted by the actual old softmax, JC3. -/
def weightedExp (z v : C → ℝ) : ℝ := ∑ c, probability z c * Real.exp (v c)

/-- The moment is the ratio of the two actual softmax partition sums, JC3. -/
theorem weightedExp_eq_ratio (z v : C → ℝ) :
    weightedExp z v = (∑ c, Real.exp (z c + v c)) / ∑ c, Real.exp (z c) := by
  unfold weightedExp probability
  rw [sum_div]
  apply sum_congr rfl
  intro c hc
  rw [Real.exp_add]
  ring

/-- A target supplies the nonempty class witness; this moment is positive, JC3. -/
theorem weightedExp_pos (z v : C → ℝ) (t : C) : 0 < weightedExp z v := by
  rw [weightedExp_eq_ratio]
  exact div_pos (sum_exp_pos _ t) (sum_exp_pos z t)

/-- The zero increment has unit moment under the actual old probabilities, JC3. -/
theorem weightedExp_zero (z : C → ℝ) (t : C) : weightedExp z (fun _ => 0) = 1 := by
  simp only [weightedExp, Real.exp_zero, mul_one]
  exact sum_probability z t

/-- A common change in all logits cancels from the relative moment, JC3. -/
theorem weightedExp_sub_const (z v : C → ℝ) (r : ℝ) :
    weightedExp z (fun c => v c - r) = weightedExp z v / Real.exp r := by
  simp only [weightedExp, Real.exp_sub, sum_div]
  apply sum_congr rfl
  intro c hc
  ring

/-- Actual CE change under an arbitrary finite logit increment, JC3. -/
theorem crossEntropy_change_weighted (z v : C → ℝ) (t : C) :
    crossEntropy (fun c => z c + v c) t - crossEntropy z t =
      Real.log (weightedExp z v) - v t := by
  rw [weightedExp_eq_ratio, Real.log_div
    (ne_of_gt (sum_exp_pos _ t)) (ne_of_gt (sum_exp_pos z t))]
  unfold crossEntropy
  ring

/-- **Loss change is the log old-probability-weighted exponential of
relative gap changes**, JC3. The target's own relative change is zero. -/
theorem crossEntropy_relative_change (z z' : C → ℝ) (t : C) :
    crossEntropy z' t - crossEntropy z t =
      Real.log (weightedExp z (fun c => (z' c - z' t) - (z c - z t))) := by
  have hi : (fun c => z c + (z' c - z c)) = z' := by funext c; ring
  have hg : (fun c => (z' c - z' t) - (z c - z t)) =
      (fun c => (z' c - z c) - (z' t - z t)) := by funext c; ring
  have h := crossEntropy_change_weighted z (fun c => z' c - z c) t
  rw [hi] at h
  rw [hg, weightedExp_sub_const, Real.log_div
    (ne_of_gt (weightedExp_pos z _ t)) (ne_of_gt (Real.exp_pos _)), Real.log_exp]
  exact h

/-- Classwise upper increments give an upper exponential moment, JC3. -/
theorem weightedExp_mono (z v U : C → ℝ) (h : ∀ c, v c ≤ U c) :
    weightedExp z v ≤ weightedExp z U := by
  apply sum_le_sum
  intro c hc
  exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (h c)) (probability_pos z c).le

/-- Relative gap-change bounds control this query's actual loss change,
JC3. Unlike an average margin, every competitor and its old mass is retained. -/
theorem loss_change_le_relative_bound (z z' U : C → ℝ) (t : C)
    (h : ∀ c, (z' c - z' t) - (z c - z t) ≤ U c) :
    crossEntropy z' t - crossEntropy z t ≤ Real.log (weightedExp z U) := by
  rw [crossEntropy_relative_change]
  exact Real.log_le_log (weightedExp_pos z _ t) (weightedExp_mono z _ U h)

/-- A computable upper exponential moment at most one ensures loss does
not increase on this query, JC3. No new loss is assumed small. -/
theorem loss_nonincrease_of_relative_bound (z z' U : C → ℝ) (t : C)
    (h : ∀ c, (z' c - z' t) - (z c - z t) ≤ U c)
    (hU : weightedExp z U ≤ 1) : crossEntropy z' t ≤ crossEntropy z t := by
  have hl := loss_change_le_relative_bound z z' U t h
  have hu := Real.log_le_log (weightedExp_pos z U t) hU
  rw [Real.log_one] at hu
  linarith

example := weightedExp_mono (fun c : Bool => if c then (1 : ℝ) else 0)
  (fun _ => -1) (fun _ => 1) (by intro c; norm_num)
example := loss_change_le_relative_bound (fun _ : Bool => (0 : ℝ))
  (fun c => if c then 1 else 0) (fun _ => 0) true
  (by intro c; cases c <;> norm_num)
example := loss_nonincrease_of_relative_bound (fun _ : Bool => (0 : ℝ))
  (fun c => if c then 1 else 0) (fun _ => 0) true
  (by intro c; cases c <;> norm_num) (by rw [weightedExp_zero _ true])

end Transformer.Grokking.GradientEvidence
