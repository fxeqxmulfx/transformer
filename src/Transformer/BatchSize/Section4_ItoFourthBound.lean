/-
# Fourth moment of finite adapted Brownian sums

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The sharp scalar bound is three times the fourth power of the amplitude
bound times elapsed time squared, uniformly over all grid refinements.
-/

import Transformer.BatchSize.Section4_ItoMomentBounds

open MeasureTheory
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- A bounded adapted Brownian sum satisfies the fourth-moment
estimate needed for Kolmogorov continuity, Section 4.3 (2)--(3).
Coefficients may depend on every past coordinate and need not be deterministic. -/
theorem brownianItoSum_fourthMoment_bound {d : ℕ} (k : Fin d) (t : ℕ → ℝ≥0)
    (hmono : Monotone t) (H : ℕ → BrownianSample d → ℝ)
    (hH : ∀ j, StronglyMeasurable[brownianFiltration d (t j)] (H j))
    (A : ℝ≥0) (hbound : ∀ j ω, |H j ω| ≤ A) (n : ℕ) :
    (∫ ω, brownianItoSum k t H n ω ^ 4 ∂brownianNoiseLaw d) ≤
      3 * A ^ 4 * ((t n : ℝ) - t 0) ^ 2 := by
  induction n with
  | zero => simp [brownianItoSum]
  | succ n ih =>
    let Z : BrownianSample d → ℝ := brownianItoSum k t H n
    let δ : ℝ := (t (n + 1) : ℝ) - t n
    let u : ℝ := (t n : ℝ) - t 0
    have hδ : 0 ≤ δ := sub_nonneg.mpr (NNReal.coe_le_coe.mpr (hmono (Nat.le_succ n)))
    have hZ := brownianItoSum_adapted k t hmono H hH n
    have hZL4 := brownianItoSum_memLp_four k t H hH A hbound n
    have hstep := brownianIncrement_fourth_energy_bound k Z (H n) (t n) (t (n + 1))
      (hmono (Nat.le_succ n)) hZ (hH n) hZL4 A (hbound n)
    have hsecond := brownianItoSum_secondMoment_bound k t hmono H hH A hbound n
    have heq (ω : BrownianSample d) : brownianItoSum k t H (n + 1) ω =
        Z ω + H n ω * brownianIncrement k (t n) (t (n + 1)) ω := by
      simp only [brownianItoSum, Finset.sum_range_succ, Z]
    simp_rw [heq]
    calc
      _ ≤ (∫ ω, Z ω ^ 4 ∂brownianNoiseLaw d) +
          6 * A ^ 2 * δ * (∫ ω, Z ω ^ 2 ∂brownianNoiseLaw d) + 3 * A ^ 4 * δ ^ 2 := hstep
      _ ≤ 3 * A ^ 4 * u ^ 2 + 6 * A ^ 2 * δ * (A ^ 2 * u) + 3 * A ^ 4 * δ ^ 2 := by
        exact add_le_add (add_le_add ih
          (mul_le_mul_of_nonneg_left hsecond (by positivity))) le_rfl
      _ = _ := by dsimp [δ, u]; ring

/-- Joint nonvacuity of the fourth-moment estimate, Section 4.3:
unit amplitudes on an increasing half-unit grid. -/
example : Monotone (fun j : ℕ => (j : ℝ≥0) / 2) ∧
    (∀ j : ℕ, StronglyMeasurable[brownianFiltration 1 ((j : ℝ≥0) / 2)]
      ((fun _ : ℕ => fun _ : BrownianSample 1 => (1 : ℝ)) j)) ∧
    (∀ (j : ℕ) (ω : BrownianSample 1),
      |(fun _ : ℕ => fun _ : BrownianSample 1 => (1 : ℝ)) j ω| ≤ (1 : ℝ≥0)) := by
  refine ⟨?_, fun _ => stronglyMeasurable_const, by simp⟩
  intro i j hij
  exact div_le_div_of_nonneg_right (by exact_mod_cast hij) (by norm_num)

end Transformer.BatchSize
