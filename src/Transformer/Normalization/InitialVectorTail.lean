/-
# Dimension-independent concentration of bounded vector sums

The vector Hoeffding input in Appendix B, proof of Theorem 4.2 of
arXiv:2510.22026v2.
-/

import Transformer.Normalization.InitialVectorMoments
import Transformer.Normalization.InitialConcentration

open scoped BigOperators
open MeasureTheory

namespace Transformer.Normalization

variable {X E : Type*} [NormedAddCommGroup E]

/-- A summand of norm at most `3` makes the norm of the sum change by at
most `6` when one coordinate is changed. Source: arXiv:2510.22026v2,
Appendix B, the vector concentration step in Theorem 4.2. -/
theorem coordinateOscillation_norm_sum (f : X → E) (hb : ∀ a, ‖f a‖ ≤ 3) (n : ℕ) :
    CoordinateOscillation (fun x : Fin n → X => ‖∑ i, f (x i)‖) 6 := by
  intro x i a
  have heq : (∑ k, f (Function.update x i a k)) - (∑ k, f (x k)) = f a - f (x i) := by
    rw [← Finset.sum_sub_distrib,
      Finset.sum_eq_single_of_mem i (Finset.mem_univ i)]
    · simp
    · intro k _ hki
      simp [Function.update_of_ne hki]
  apply (abs_norm_sub_norm_le _ _).trans
  rw [heq]
  exact (norm_sub_le _ _).trans (by linarith [hb a, hb (x i)])

/-- Zero vectors satisfy the summand bound. -/
example : ∀ _ : Unit, ‖(0 : ℝ)‖ ≤ 3 := by simp

variable [TopologicalSpace X] [CompactSpace X] [Nonempty X]
  [MeasurableSpace X] [BorelSpace X] [SecondCountableTopology X]
  [FirstCountableTopology X] [LocallyCompactSpace X]
  [InnerProductSpace ℝ E] [CompleteSpace E]

/-- A bounded independent vector sum exceeds its bias and square-root
fluctuation by `t` with probability at most `exp(-t²/(72 n))`.
Source: arXiv:2510.22026v2, Appendix B, vector Hoeffding in Theorem 4.2. -/
theorem measureReal_pi_norm_sum_gt_le (μ : Measure X) [IsProbabilityMeasure μ]
    (f : X → E) (hf : Continuous f) (hb : ∀ a, ‖f a‖ ≤ 3)
    (n : ℕ) (t : ℝ) (ht : 0 ≤ t) :
    (Measure.pi (fun _ : Fin n => μ)).real {x : Fin n → X |
      (n : ℝ) * ‖∫ a, f a ∂μ‖ + 6 * Real.sqrt n + t < ‖∑ i, f (x i)‖} ≤
      Real.exp (-t ^ 2 / (72 * (n : ℝ))) := by
  let F : (Fin n → X) → ℝ := fun x => ‖∑ i, f (x i)‖
  have hF : Continuous F := by dsimp [F]; fun_prop
  let c : NNReal := ⟨36 * (n : ℝ), by positivity⟩
  have hsg : ProbabilityTheory.HasSubgaussianMGF
      (fun x => F x - ∫ y, F y ∂Measure.pi (fun _ : Fin n => μ)) c
      (Measure.pi (fun _ : Fin n => μ)) := by
    constructor
    · intro s
      exact Perspective.integrable_of_continuous_compact (by fun_prop) _
    · intro s
      have h := mgf_centered_pi_le μ 6 (by norm_num) n F hF
        (coordinateOscillation_norm_sum f hb n) s
      change (∫ x, Real.exp (s * (F x - ∫ y, F y ∂Measure.pi (fun _ : Fin n => μ)))
        ∂Measure.pi (fun _ : Fin n => μ)) ≤ Real.exp ((36 * (n : ℝ)) * s ^ 2 / 2)
      simpa only [ProbabilityTheory.mgf,
        show (6 : ℝ) ^ 2 = 36 by norm_num, mul_comm (n : ℝ) 36] using h
  have hmean := integral_pi_norm_sum_le μ f hf hb n
  have hsub : {x : Fin n → X |
      (n : ℝ) * ‖∫ a, f a ∂μ‖ + 6 * Real.sqrt n + t < ‖∑ i, f (x i)‖} ⊆
      {x | t ≤ F x - ∫ y, F y ∂Measure.pi (fun _ : Fin n => μ)} := by
    intro x hx
    change t ≤ F x - ∫ y, F y ∂Measure.pi (fun _ : Fin n => μ)
    change _ < F x at hx
    change (∫ y, F y ∂Measure.pi (fun _ : Fin n => μ)) ≤ _ at hmean
    linarith
  apply (measureReal_mono (μ := Measure.pi (fun _ : Fin n => μ)) hsub).trans
  have htail := hsg.measure_ge_le ht
  change (Measure.pi (fun _ : Fin n => μ)).real
    {x | t ≤ F x - ∫ y, F y ∂Measure.pi (fun _ : Fin n => μ)} ≤
      Real.exp (-t ^ 2 / (2 * (36 * (n : ℝ)))) at htail
  simpa only [show (2 : ℝ) * (36 * (n : ℝ)) = 72 * (n : ℝ) by ring] using htail

/-- Constant zero summands and a nonnegative threshold realize the hypotheses. -/
example : Continuous (fun _ : Unit => (0 : ℝ)) ∧
    (∀ _ : Unit, ‖(0 : ℝ)‖ ≤ 3) ∧ (0 : ℝ) ≤ 1 := by simp [continuous_const]

end Transformer.Normalization
