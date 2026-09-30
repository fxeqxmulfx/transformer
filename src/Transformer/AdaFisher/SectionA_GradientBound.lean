/-
# AdaFisher: the corrected last step of the stochastic argument

arXiv:2405.16397v3, Appendix A.2, `eq:bound_2`.
An explicit weighted-energy budget implies a small gradient expectation.
This conditional statement does not assume the false β=1 version of
Proposition 3.4 or conceal its stochastic estimate in a structure field.
-/

import Transformer.AdaFisher.SectionA_StochasticTerms
import Mathlib.Data.Finset.Max

open scoped BigOperators

noncomputable section

namespace Transformer.AdaFisher

/-- Correct extraction of a small expected gradient from the weighted
budget in `eq:bound_2`, Appendix A.2. `E t` is the expected squared
gradient at positive time t+1. The omitted η remains in the denominator:
`min E ≤ L*K/(η*sqrt(T))`. The budget is an explicit premise; this theorem
does not claim to establish the preceding Adam-type stochastic estimate. -/
theorem gradient_bound_of_budget (η L K : ℝ) (E : ℕ → ℝ) (T : ℕ)
    (hη : 0 < η) (hL : 0 < L) (hT : 0 < T) (hE : ∀ t, 0 ≤ E t)
    (hbudget : (∑ t ∈ Finset.range T, stepSize η t / L * E t) ≤ K) :
    ∃ t ∈ Finset.range T, E t ≤ L * K / (η * Real.sqrt T) := by
  obtain ⟨t, ht, hmin⟩ := Finset.exists_min_image (Finset.range T) E
    ⟨0, Finset.mem_range.mpr hT⟩
  have hsT : 0 < Real.sqrt (T : ℝ) := Real.sqrt_pos.mpr (by exact_mod_cast hT)
  have hweight (u : ℕ) (hu : u ∈ Finset.range T) :
      η / Real.sqrt T / L ≤ stepSize η u / L := by
    unfold stepSize
    apply div_le_div_of_nonneg_right _ hL.le
    apply div_le_div_of_nonneg_left hη.le
      (Real.sqrt_pos.mpr (by positivity))
    apply Real.sqrt_le_sqrt
    exact_mod_cast Nat.succ_le_of_lt (Finset.mem_range.mp hu)
  have hsum : (T : ℝ) * (η / Real.sqrt T / L) * E t ≤ K := by
    calc
      _ = ∑ u ∈ Finset.range T, η / Real.sqrt T / L * E t := by simp [mul_assoc]
      _ ≤ ∑ u ∈ Finset.range T, stepSize η u / L * E u := by
        apply Finset.sum_le_sum
        intro u hu
        exact mul_le_mul (hweight u hu) (hmin u hu) (hE t)
          (div_nonneg (div_nonneg hη.le (Real.sqrt_nonneg _)) hL.le)
      _ ≤ K := hbudget
  have hTsqrt := Real.sq_sqrt (Nat.cast_nonneg T : (0 : ℝ) ≤ T)
  have heq : (T : ℝ) * (η / Real.sqrt T / L) = η * Real.sqrt T / L := by
    have hs0 : Real.sqrt (T : ℝ) ≠ 0 := ne_of_gt hsT
    field_simp
    nlinarith
  rw [heq] at hsum
  have hcoef : 0 < η * Real.sqrt T / L := div_pos (mul_pos hη hsT) hL
  have hb := (le_div_iff₀ hcoef).mpr (by simpa [mul_comm] using hsum)
  refine ⟨t, ht, ?_⟩
  convert hb using 1
  simp only [div_eq_mul_inv, mul_inv_rev, inv_inv]
  ring

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (0 : ℕ) < 1 ∧
    (∀ t, 0 ≤ (fun _ : ℕ => (0 : ℝ)) t) ∧
    (∑ t ∈ Finset.range 1, stepSize 1 t / 2 * (fun _ : ℕ => (0 : ℝ)) t) ≤ 0 := by
  norm_num

end Transformer.AdaFisher
