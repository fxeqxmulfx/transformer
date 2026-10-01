/-
# Actual weighted test expectations under mean-square convergence

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The random bounded past weight remains fixed as the actual state converges.
-/

import Transformer.BatchSize.Section4_BoundedObservableLimits

open MeasureTheory Filter
open scoped Topology

noncomputable section

namespace Transformer.BatchSize

/-- Genuine mean-square convergence implies convergence of
normalized C2 test expectations with any fixed measurable bounded
weight, Section 4.3 (2)--(3). -/
theorem weighted_test_meanSquare_tendsto {Ω : Type*} [MeasurableSpace Ω] {d : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (X : ℕ → Ω → EucSpace d)
    (hX : ∀ m, MemLp (X m) 2 P) (Y : Ω → EucSpace d) (hY : MemLp Y 2 P)
    (hlim : Tendsto (fun m => ∫ ω, ‖X m ω - Y ω‖ ^ 2 ∂P) atTop (𝓝 0))
    (F : Ω → ℝ) (hFm : Measurable F) (hF1 : ∀ ω, |F ω| ≤ 1)
    (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) :
    Tendsto (fun m => ∫ ω, F ω * φ (X m ω) ∂P) atTop (𝓝 (∫ ω, F ω * φ (Y ω) ∂P)) := by
  apply integral_bounded_observable_tendsto P X Y (meanSquare_tendstoInMeasure P X hX Y hY hlim)
    (fun ω x => F ω * φ x) (fun _ => continuous_const.mul hφ.1.continuous)
    (fun m => hFm.aestronglyMeasurable.mul (hφ.1.continuous.comp_aestronglyMeasurable
      (hX m).aestronglyMeasurable)) 1
  intro ω y
  have hφ1 : |φ y| ≤ 1 := by
    simpa only [norm_iteratedFDeriv_zero, Real.norm_eq_abs] using hφ.2 0 (by norm_num) y
  rw [abs_mul]
  exact (mul_le_mul (hF1 ω) hφ1 (abs_nonneg _) (by norm_num)).trans_eq (one_mul _)

/-- Joint nonvacuity of weighted test-limit hypotheses,
Section 4.3: a nonzero deterministic state sequence, a nonzero
bounded weight and a normalized nonzero smooth test. -/
example : MemLp (fun _ : BrownianSample 1 => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) 2
    (brownianNoiseLaw 1) ∧
    Measurable (fun _ : BrownianSample 1 => (1 : ℝ)) ∧ |(1 : ℝ)| ≤ 1 ∧
    BoundedSmoothTest 2 (fun _ : EucSpace 1 => (1 : ℝ)) := by
  refine ⟨memLp_const _, measurable_const, by norm_num, contDiff_const, ?_⟩
  intro j hj y
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
