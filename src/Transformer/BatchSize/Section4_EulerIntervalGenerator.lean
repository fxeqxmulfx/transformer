/-
# The generator identity in physical time on an actual Euler interval

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The frozen Brownian identity is transported to the actual grid interval
and to the full Brownian interpolation, with any bounded past weight.
-/

import Transformer.BatchSize.Section4_EulerIntervals

open MeasureTheory Filter
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The actual Euler increment and its full interpolated generator
satisfy the weighted identity on each physical grid interval,
Section 4.3 (2)--(3). The grid may have zero-length intervals. -/
theorem eulerPath_interval_generator_identity {d : ℕ}
    (b : EucSpace d → EucSpace d) (a : EucSpace d → Fin d → ℝ)
    (hb : Continuous b) (ha : ∀ k, Continuous (fun x => a x k))
    (M A : NNReal) (hbM : ∀ x, ‖b x‖ ≤ M) (haA : ∀ x k, |a x k| ≤ A)
    (t : ℕ → NNReal) (hmono : Monotone t) (x₀ : EucSpace d) (N j : ℕ) (hj : j < N)
    (F : BrownianSample d → ℝ) (hF : StronglyMeasurable[brownianFiltration d (t j)] F)
    (hF1 : ∀ ω, |F ω| ≤ 1) (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) :
    (∫ ω, F ω * (φ (eulerChain b a t x₀ (j + 1) ω) - φ (eulerChain b a t x₀ j ω))
      ∂brownianNoiseLaw d) =
      ∫ u in (t j : ℝ)..(t (j + 1) : ℝ), ∫ ω, F ω * frozenGaussianGenerator
        (b (eulerChain b a t x₀ j ω)) (a (eulerChain b a t x₀ j ω)) φ
        (eulerPathValue b a t x₀ N ω u) ∂brownianNoiseLaw d := by
  have hst := hmono (Nat.le_succ j)
  have h := frozenBrownianState_generator_identity b a hb ha M A hbM haA
    (t j) (t (j + 1)) hst (eulerChain b a t x₀ j)
    (eulerChain_adapted b a hb ha t hmono x₀ j) F hF hF1 φ hφ
  simp_rw [← eulerChain_succ_eq_frozenBrownianState] at h
  rw [h]
  let G (u : ℝ) := ∫ ω, F ω * frozenGaussianGenerator
    (b (eulerChain b a t x₀ j ω)) (a (eulerChain b a t x₀ j ω)) φ
    (eulerPathValue b a t x₀ N ω u) ∂brownianNoiseLaw d
  have hshift : (∫ v in (0 : ℝ)..((t (j + 1) : ℝ) - t j), G ((t j : ℝ) + v)) =
      ∫ u in (t j : ℝ)..(t (j + 1) : ℝ), G u := by
    rw [intervalIntegral.integral_comp_add_left]
    simp only [add_zero, add_sub_cancel]
  rw [← hshift]
  apply intervalIntegral.integral_congr
  intro v hv
  rw [Set.uIcc_of_le (sub_nonneg.mpr (NNReal.coe_le_coe.mpr hst))] at hv
  have hv0 : 0 ≤ v := hv.1
  have hl : t j ≤ t j + v.toNNReal := le_add_of_nonneg_right zero_le
  have hr : t j + v.toNNReal ≤ t (j + 1) := by
    apply NNReal.coe_le_coe.mp
    rw [NNReal.coe_add, Real.coe_toNNReal v hv0]
    linarith [hv.2]
  apply integral_congr_ae
  exact Eventually.of_forall fun ω => by
    dsimp only [G]
    have hpath := eulerPathValue_eq_frozenBrownianState b a t hmono x₀ N j hj
      (t j + v.toNNReal) hl hr ω
    have hreal : ((t j + v.toNNReal : NNReal) : ℝ) = (t j : ℝ) + v := by
      rw [NNReal.coe_add, Real.coe_toNNReal v hv0]
    rw [← hreal, hpath]

/-- Joint nonvacuity of the interval generator hypotheses,
Section 4.3: constant bounded coefficients, increasing unit times,
a nonzero past weight and a normalized nonzero C2 test. -/
example : Continuous (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    (∀ k : Fin 1, Continuous (fun _ : EucSpace 1 => (k : ℝ) + 1)) ∧
    (∀ x : EucSpace 1, ‖(fun _ : EucSpace 1 => (0 : EucSpace 1)) x‖ ≤ (1 : NNReal)) ∧
    (∀ k : Fin 1, |(k : ℝ) + 1| ≤ (1 : NNReal)) ∧
    Monotone (fun n : ℕ => (n : NNReal)) ∧ (1 : ℕ) < 3 ∧
    StronglyMeasurable[brownianFiltration 1 1] (fun _ : BrownianSample 1 => (1 : ℝ)) ∧
    BoundedSmoothTest 2 (fun _ : EucSpace 1 => (1 : ℝ)) := by
  refine ⟨continuous_const, fun _ => continuous_const, fun _ => by simp,
    (fun k => by fin_cases k; norm_num),
    (fun i j hij => by change (i : NNReal) ≤ (j : NNReal); exact_mod_cast hij),
    by omega, stronglyMeasurable_const, contDiff_const, ?_⟩
  intro j hj y
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
