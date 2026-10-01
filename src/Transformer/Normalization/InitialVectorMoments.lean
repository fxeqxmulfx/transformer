/-
# Second moments of independent vector sums

The mean-norm input for vector concentration in Appendix B, proof of
Theorem 4.2 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.InitialProduct
import Mathlib.Analysis.InnerProductSpace.LinearMap

open scoped BigOperators
open MeasureTheory

namespace Transformer.Normalization

variable {X E : Type*} [TopologicalSpace X] [CompactSpace X]
  [MeasurableSpace X] [BorelSpace X] [SecondCountableTopology X]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

omit [CompleteSpace E] in
/-- The mean of a sum of independent coordinates is the sum of the means.
Source: arXiv:2510.22026v2, Appendix B, centered vector sums. -/
theorem integral_pi_sum (μ : Measure X) [IsProbabilityMeasure μ]
    (f : X → E) (hf : Continuous f) (n : ℕ) :
    (∫ x : Fin n → X, ∑ i, f (x i) ∂Measure.pi (fun _ : Fin n => μ)) =
      n • ∫ a, f a ∂μ := by
  rw [integral_finsetSum Finset.univ (f := fun i (x : Fin n → X) => f (x i)) (fun i _ =>
    Perspective.integrable_of_continuous_compact (hf.comp (continuous_apply i)) _)]
  have hi : ∀ i : Fin n,
      (∫ x : Fin n → X, f (x i) ∂Measure.pi (fun _ : Fin n => μ)) = ∫ a, f a ∂μ := by
    intro i
    have h := integral_map (μ := Measure.pi (fun _ : Fin n => μ))
      (measurable_pi_apply i).aemeasurable hf.aestronglyMeasurable
    rw [(measurePreserving_eval (μ := fun _ : Fin n => μ) i).map_eq] at h
    exact h.symm
  simp only [hi, Finset.sum_const, Finset.card_univ, Fintype.card_fin]

/-- Constant functions meet the continuity hypothesis. -/
example : Continuous (fun _ : Unit => (0 : ℝ)) := continuous_const

/-- Cross terms vanish for centered independent vectors. Source:
arXiv:2510.22026v2, Appendix B, vector Hoeffding step in Theorem 4.2. -/
theorem integral_pi_sum_norm_sq (μ : Measure X) [IsProbabilityMeasure μ]
    (f : X → E) (hf : Continuous f) (hmean : (∫ a, f a ∂μ) = 0) (n : ℕ) :
    (∫ x : Fin n → X, ‖∑ i, f (x i)‖ ^ 2 ∂Measure.pi (fun _ : Fin n => μ)) =
      (n : ℝ) * ∫ a, ‖f a‖ ^ 2 ∂μ := by
  induction n with
  | zero => simp
  | succ n ih =>
    have hS : Continuous (fun x : Fin n → X => ∑ i, f (x i)) := by fun_prop
    have hSmean : (∫ x : Fin n → X, ∑ i, f (x i) ∂Measure.pi (fun _ : Fin n => μ)) = 0 := by
      rw [integral_pi_sum μ f hf n, hmean, smul_zero]
    rw [integral_pi_cons μ n _ (by fun_prop)]
    have hs : ∀ a : X,
        (∫ x : Fin n → X, ‖∑ i : Fin (n + 1), f ((Fin.cons a x : Fin (n + 1) → X) i)‖ ^ 2
          ∂Measure.pi (fun _ : Fin n => μ)) =
        ‖f a‖ ^ 2 + (n : ℝ) * ∫ b, ‖f b‖ ^ 2 ∂μ := by
      intro a
      simp only [Fin.sum_univ_succ, Fin.cons_zero, Fin.cons_succ, norm_add_sq_real]
      rw [integral_add
        (Perspective.integrable_of_continuous_compact (by fun_prop) _)
        (Perspective.integrable_of_continuous_compact (by fun_prop) _),
        integral_add
        (Perspective.integrable_of_continuous_compact (by fun_prop) _)
        (Perspective.integrable_of_continuous_compact (by fun_prop) _),
        integral_const_mul]
      have hcross := (innerSL ℝ (f a)).integral_comp_comm
        (Perspective.integrable_of_continuous_compact hS (Measure.pi (fun _ : Fin n => μ)))
      simp only [innerSL_apply_apply, hSmean, inner_zero_right] at hcross
      rw [hcross, mul_zero, integral_const, probReal_univ, one_smul, add_zero, ih]
    simp_rw [hs]
    rw [integral_add (Perspective.integrable_of_continuous_compact (by fun_prop) μ)
      (integrable_const _), integral_const, probReal_univ, one_smul]
    push_cast
    ring

/-- A zero vector-valued function is centered and continuous. -/
example : Continuous (fun _ : Unit => (0 : ℝ)) ∧
    (∫ _ : Unit, (0 : ℝ) ∂Measure.dirac ()) = 0 := by simp [continuous_const]

omit [SecondCountableTopology X] in
/-- A scalar second moment controls the square of its mean. Source:
arXiv:2510.22026v2, Appendix B, the mean norm of the centered vector sum. -/
theorem integral_sq_mean_le (μ : Measure X) [IsProbabilityMeasure μ]
    (g : X → ℝ) (hg : Continuous g) :
    (∫ a, g a ∂μ) ^ 2 ≤ ∫ a, g a ^ 2 ∂μ := by
  let m := ∫ a, g a ∂μ
  have h : 0 ≤ ∫ a, (g a - m) ^ 2 ∂μ := integral_nonneg (fun _ => sq_nonneg _)
  have heq : (fun a => (g a - m) ^ 2) = fun a => g a ^ 2 - 2 * m * g a + m ^ 2 := by
    funext a
    ring
  rw [heq, integral_add
    (Perspective.integrable_of_continuous_compact (by fun_prop) μ) (integrable_const _),
    integral_sub (Perspective.integrable_of_continuous_compact (by fun_prop) μ)
      (Perspective.integrable_of_continuous_compact (by fun_prop) μ),
    integral_const_mul, integral_const, probReal_univ, one_smul] at h
  change m ^ 2 ≤ _
  change 0 ≤ (∫ a, g a ^ 2 ∂μ) - 2 * m * m + m ^ 2 at h
  nlinarith

/-- Constant functions meet the continuity hypothesis. -/
example : Continuous (fun _ : Unit => (0 : ℝ)) := continuous_const

/-- Bounded independent vectors have mean norm at most their bias plus
`6 sqrt n`. Source: arXiv:2510.22026v2, Appendix B, vector concentration
in the proof of Theorem 4.2; the bound `3` covers the exponential tilt. -/
theorem integral_pi_norm_sum_le (μ : Measure X) [IsProbabilityMeasure μ]
    (f : X → E) (hf : Continuous f) (hb : ∀ a, ‖f a‖ ≤ 3) (n : ℕ) :
    (∫ x : Fin n → X, ‖∑ i, f (x i)‖ ∂Measure.pi (fun _ : Fin n => μ)) ≤
      (n : ℝ) * ‖∫ a, f a ∂μ‖ + 6 * Real.sqrt n := by
  let m := ∫ a, f a ∂μ
  let g : X → E := fun a => f a - m
  have hg : Continuous g := hf.sub continuous_const
  have hm : ‖m‖ ≤ 3 := by
    simpa only [m, probReal_univ, mul_one] using
      norm_integral_le_of_norm_le_const (ae_of_all μ hb)
  have hgmean : (∫ a, g a ∂μ) = 0 := by
    dsimp [g]
    rw [integral_sub (Perspective.integrable_of_continuous_compact hf μ) (integrable_const m),
      integral_const, probReal_univ, one_smul]
    exact sub_self m
  have hgb : ∀ a, ‖g a‖ ≤ 6 := fun a => (norm_sub_le _ _).trans (by linarith [hb a])
  have hgsq : (∫ a, ‖g a‖ ^ 2 ∂μ) ≤ 36 := by
    calc _ ≤ ∫ _ : X, (36 : ℝ) ∂μ :=
        integral_mono (Perspective.integrable_of_continuous_compact (by fun_prop) μ)
          (integrable_const _) (fun a => by nlinarith [hgb a, norm_nonneg (g a)])
      _ = 36 := by simp
  let S : (Fin n → X) → E := fun x => ∑ i, g (x i)
  have hS : Continuous S := by dsimp [S]; fun_prop
  have hsq : (∫ x, ‖S x‖ ∂Measure.pi (fun _ : Fin n => μ)) ^ 2 ≤ 36 * (n : ℝ) := by
    apply (integral_sq_mean_le _ _ hS.norm).trans
    change (∫ x : Fin n → X, ‖∑ i, g (x i)‖ ^ 2 ∂Measure.pi (fun _ : Fin n => μ)) ≤ _
    rw [integral_pi_sum_norm_sq μ g hg hgmean n]
    nlinarith [mul_le_mul_of_nonneg_left hgsq (Nat.cast_nonneg (α := ℝ) n)]
  have hSnorm : (∫ x, ‖S x‖ ∂Measure.pi (fun _ : Fin n => μ)) ≤ 6 * Real.sqrt n := by
    nlinarith [Real.sq_sqrt (Nat.cast_nonneg (α := ℝ) n), Real.sqrt_nonneg (n : ℝ)]
  have hpoint : ∀ x : Fin n → X, ‖∑ i, f (x i)‖ ≤ ‖S x‖ + (n : ℝ) * ‖m‖ := by
    intro x
    have heq : (∑ i, f (x i)) = S x + n • m := by
      dsimp [S, g]
      rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin]
      abel
    rw [heq]
    apply (norm_add_le _ _).trans
    rw [← Nat.cast_smul_eq_nsmul ℝ, norm_smul, Real.norm_eq_abs,
      abs_of_nonneg (Nat.cast_nonneg (α := ℝ) n)]
  calc _ ≤ ∫ x : Fin n → X, ‖S x‖ + (n : ℝ) * ‖m‖ ∂Measure.pi (fun _ : Fin n => μ) :=
      integral_mono (Perspective.integrable_of_continuous_compact (by fun_prop) _)
        (Perspective.integrable_of_continuous_compact (by fun_prop) _) hpoint
    _ = (∫ x, ‖S x‖ ∂Measure.pi (fun _ : Fin n => μ)) + (n : ℝ) * ‖m‖ := by
      rw [integral_add (Perspective.integrable_of_continuous_compact hS.norm _) (integrable_const _)]
      simp
    _ ≤ _ := by linarith

/-- Zero vectors meet the continuity and boundedness hypotheses. -/
example : Continuous (fun _ : Unit => (0 : ℝ)) ∧ (∀ _ : Unit, ‖(0 : ℝ)‖ ≤ 3) := by
  simp [continuous_const]

end Transformer.Normalization
