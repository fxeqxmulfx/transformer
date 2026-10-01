/-
# The actual finite Brownian Euler martingale identity

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The physical interval identities telescope for any bounded observable
measurable in the past at the chosen starting node.
-/

import Transformer.BatchSize.Section4_DyadicGenerator
import Transformer.BatchSize.Section4_EulerIntervalGenerator

open MeasureTheory Filter
open scoped NNReal BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- The actual stopped dyadic Euler process satisfies the weighted
integrated frozen-generator identity on any tail of its grid,
Section 4.3 (2)--(3). Integrability is proved from actual uniform
coefficient and C2 test bounds before the intervals are telescoped. -/
theorem dyadicEuler_weighted_generator_identity {d : ℕ}
    (b : EucSpace d → EucSpace d) (a : EucSpace d → Fin d → ℝ)
    (hb : Continuous b) (ha : ∀ k, Continuous (fun x => a x k))
    (M A : NNReal) (hbM : ∀ x, ‖b x‖ ≤ M) (haA : ∀ x k, |a x k| ≤ A)
    (x₀ : EucSpace d) (T : NNReal) (m J : ℕ) (hJ : J ≤ dyadicEndpointIndex T m)
    (F : BrownianSample d → ℝ)
    (hF : StronglyMeasurable[brownianFiltration d (dyadicGrid T m J)] F)
    (hF1 : ∀ ω, |F ω| ≤ 1) (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) :
    (∫ ω, F ω * (φ (dyadicEuler b a x₀ T m ω) -
      φ (eulerChain b a (dyadicGrid T m) x₀ J ω)) ∂brownianNoiseLaw d) =
      ∫ u in (dyadicGrid T m J : ℝ)..(T : ℝ), ∫ ω,
        F ω * dyadicGeneratorValue b a x₀ T m φ u ω ∂brownianNoiseLaw d := by
  let t := dyadicGrid T m
  let N := dyadicEndpointIndex T m
  let X := eulerChain b a t x₀
  let E (j : ℕ) := ∫ ω, F ω * φ (X j ω) ∂brownianNoiseLaw d
  let G (u : ℝ) := ∫ ω, F ω * dyadicGeneratorValue b a x₀ T m φ u ω ∂brownianNoiseLaw d
  have hFm := (hF.mono ((brownianFiltration d).le _)).measurable
  have hφ1 (x : EucSpace d) : |φ x| ≤ 1 := by
    simpa only [norm_iteratedFDeriv_zero, Real.norm_eq_abs] using hφ.2 0 (by norm_num) x
  have hEi (j : ℕ) : Integrable (fun ω => F ω * φ (X j ω)) (brownianNoiseLaw d) :=
    (integrable_const (1 : ℝ)).mono'
      (hFm.mul (hφ.1.continuous.measurable.comp
        ((eulerChain_adapted b a hb ha t (dyadicGrid_monotone T m) x₀ j).mono
          ((brownianFiltration d).le _)).measurable)).aestronglyMeasurable
      (Eventually.of_forall fun ω => by
        rw [Real.norm_eq_abs, abs_mul]
        exact (mul_le_mul (hF1 ω) (hφ1 _) (abs_nonneg _) (by norm_num)).trans_eq (one_mul _))
  have hGb := dyadicGenerator_expectation_bounded b a hb ha M A hbM haA x₀ T m φ hφ F hFm hF1
  have hGi (v w : ℝ) : IntervalIntegrable G volume v w := by
    apply intervalIntegrable_iff.mpr
    apply Measure.integrableOn_of_bounded
      (by rw [Real.volume_uIoc]; exact ENNReal.ofReal_ne_top) hGb.1.aestronglyMeasurable
    exact Eventually.of_forall fun u => by simpa only [Real.norm_eq_abs] using hGb.2 u
  have hstep (j : ℕ) (hj : j ∈ Finset.Ico J N) :
      E (j + 1) - E j = ∫ u in (t j : ℝ)..(t (j + 1) : ℝ), G u := by
    have hj' := Finset.mem_Ico.mp hj
    have hFj := hF.mono ((brownianFiltration d).mono (dyadicGrid_monotone T m hj'.1))
    have hi := eulerPath_interval_generator_identity b a hb ha M A hbM haA t
      (dyadicGrid_monotone T m) x₀ N j hj'.2 F hFj hF1 φ hφ
    simp_rw [mul_sub] at hi
    rw [integral_sub (hEi (j + 1)) (hEi j)] at hi
    rw [hi]
    apply intervalIntegral.integral_congr_ae
    filter_upwards [volume.ae_ne (t (j + 1) : ℝ)] with u hne
    intro hu
    rw [Set.uIoc_of_le (NNReal.coe_le_coe.mpr (dyadicGrid_monotone T m (Nat.le_succ j)))] at hu
    have huj : u < t (j + 1) := lt_of_le_of_ne hu.2 hne
    have hidx := dyadicLeftIndex_on_interval T m j u ⟨hu.1.le, huj⟩
    apply integral_congr_ae
    exact Eventually.of_forall fun ω => by
      dsimp only [G, dyadicGeneratorValue, dyadicLeftState]
      rw [hidx]
  have hsum := Finset.sum_congr rfl hstep
  rw [Finset.sum_Ico_sub E hJ,
    intervalIntegral.sum_integral_adjacent_intervals_Ico hJ
      (fun _ _ => hGi _ _)] at hsum
  dsimp only [t] at hsum
  rw [dyadicGrid_endpoint] at hsum
  simp_rw [mul_sub]
  change (∫ ω, F ω * φ (X N ω) - F ω * φ (X J ω) ∂brownianNoiseLaw d) = _
  rw [integral_sub (hEi N) (hEi J)]
  exact hsum

/-- Joint nonvacuity of the finite Euler martingale hypotheses,
Section 4.3: bounded constant coefficients, a nonzero past weight,
an actual initial node and a normalized nonzero C2 test. -/
example : Continuous (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    (∀ k : Fin 1, Continuous (fun _ : EucSpace 1 => (k : ℝ) + 1)) ∧
    (∀ x : EucSpace 1, ‖(fun _ : EucSpace 1 => (0 : EucSpace 1)) x‖ ≤ (1 : NNReal)) ∧
    (∀ k : Fin 1, |(k : ℝ) + 1| ≤ (1 : NNReal)) ∧
    (0 : ℕ) ≤ dyadicEndpointIndex 2 1 ∧
    StronglyMeasurable[brownianFiltration 1 0] (fun _ : BrownianSample 1 => (1 : ℝ)) ∧
    |(1 : ℝ)| ≤ 1 ∧ BoundedSmoothTest 2 (fun _ : EucSpace 1 => (1 : ℝ)) := by
  refine ⟨continuous_const, fun _ => continuous_const, fun _ => by simp,
    (fun k => by fin_cases k; norm_num), Nat.zero_le _, stronglyMeasurable_const,
    by norm_num, contDiff_const, ?_⟩
  intro j hj y
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
