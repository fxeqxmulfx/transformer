/-
# The actual frozen Brownian interval satisfies the generator identity

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The vector increment's Gaussian conditional law and the integrated
Gaussian identity give the genuine weighted Brownian expectation formula.
-/

import Transformer.BatchSize.Section4_FrozenBrownianExpectation
import Transformer.BatchSize.Section4_RandomGaussianIdentity

open MeasureTheory Filter
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The integrated generator formula on an actual frozen Brownian
Euler interval, Section 4.3 (2)--(3). The weight is any bounded
observable in the entire joint past, and the test is genuinely C2. -/
theorem frozenBrownianState_generator_identity {d : ℕ}
    (b : EucSpace d → EucSpace d) (a : EucSpace d → Fin d → ℝ)
    (hb : Continuous b) (ha : ∀ k, Continuous (fun x => a x k))
    (M S : NNReal) (hbM : ∀ x, ‖b x‖ ≤ M) (haS : ∀ x k, |a x k| ≤ S)
    (s t : ℝ≥0) (hst : s ≤ t) (X : BrownianSample d → EucSpace d)
    (hX : StronglyMeasurable[brownianFiltration d s] X) (F : BrownianSample d → ℝ)
    (hF : StronglyMeasurable[brownianFiltration d s] F) (hF1 : ∀ ω, |F ω| ≤ 1)
    (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) :
    (∫ ω, F ω * (φ (frozenBrownianState b a X s t ω) - φ (X ω)) ∂brownianNoiseLaw d) =
      ∫ u in (0 : ℝ)..((t : ℝ) - s), ∫ ω, F ω * frozenGaussianGenerator
        (b (X ω)) (a (X ω)) φ
        (frozenBrownianState b a X s (s + u.toNNReal) ω) ∂brownianNoiseLaw d := by
  have hXm := (hX.mono ((brownianFiltration d).le s)).measurable
  have hFm := (hF.mono ((brownianFiltration d).le s)).measurable
  have hφbound (y : EucSpace d) : |φ y| ≤ 1 := by
    simpa only [norm_iteratedFDeriv_zero, Real.norm_eq_abs] using hφ.2 0 (by norm_num) y
  have hΨ : Measurable (fun p : EucSpace d × EucSpace d => φ p.2 - φ p.1) :=
    (hφ.1.continuous.measurable.comp measurable_snd).sub
      (hφ.1.continuous.measurable.comp measurable_fst)
  have hΨbound (x y : EucSpace d) : |φ y - φ x| ≤ 2 := by
    calc
      _ ≤ |φ y| + |φ x| := abs_sub _ _
      _ ≤ 1 + 1 := add_le_add (hφbound y) (hφbound x)
      _ = 2 := by norm_num
  have hstep := frozenBrownianState_weighted_expectation b a hb ha s t hst X hX F hF hF1
    (fun p => φ p.2 - φ p.1) hΨ 2 hΨbound
  have hflow (ω : BrownianSample d) :
      gaussianVectorFlow (fun y => φ y - φ (X ω)) (X ω) (b (X ω)) (a (X ω)) ((t : ℝ) - s) =
        gaussianVectorFlow φ (X ω) (b (X ω)) (a (X ω)) ((t : ℝ) - s) - φ (X ω) := by
    have hφi : Integrable (fun z => φ (gaussianVectorState (X ω) (b (X ω))
        (a (X ω)) ((t : ℝ) - s) z)) (standardGaussianVectorLaw d) :=
      (integrable_const (1 : ℝ)).mono'
        (hφ.1.continuous.comp (gaussianVectorState_continuous_noise _ _ _ _)).aestronglyMeasurable
        (Eventually.of_forall fun z => by simpa only [Real.norm_eq_abs] using hφbound _)
    unfold gaussianVectorFlow
    rw [integral_sub hφi (integrable_const (φ (X ω)))]
    simp
  simp_rw [hflow] at hstep
  rw [hstep, randomGaussianFlow_generator_identity (brownianNoiseLaw d) X
    (fun ω => b (X ω)) (fun ω => a (X ω)) F hXm (hb.measurable.comp hXm)
    (fun k => (ha k).measurable.comp hXm) hFm M S (fun ω => hbM _) (fun ω k => haS _ k)
    hF1 φ hφ ((t : ℝ) - s) (sub_nonneg.mpr (NNReal.coe_le_coe.mpr hst))]
  have hgenMeas : Measurable (fun p : EucSpace d × EucSpace d =>
      frozenGaussianGenerator (b p.1) (a p.1) φ p.2) :=
    frozenGaussianGenerator_measurable_parameters _ _ _ (hb.measurable.comp measurable_fst)
      (fun k => (ha k).measurable.comp measurable_fst) measurable_snd φ hφ
  apply intervalIntegral.integral_congr
  intro u hu
  have hu0 : 0 ≤ u := by
    rw [Set.uIcc_of_le (sub_nonneg.mpr (NNReal.coe_le_coe.mpr hst))] at hu
    exact hu.1
  have heq := frozenBrownianState_weighted_expectation b a hb ha s (s + u.toNNReal)
    (by exact le_add_of_nonneg_right (show (0 : NNReal) ≤ u.toNNReal from zero_le))
    X hX F hF hF1 (fun p => frozenGaussianGenerator (b p.1) (a p.1) φ p.2) hgenMeas
    (M + (d : ℝ) * (S : ℝ) ^ 2 / 2)
    (fun x y => frozenGaussianGenerator_uniform_bound (b x) (a x) M S (hbM x) (haS x) φ hφ y)
  have htime : ((s + u.toNNReal : NNReal) : ℝ) - s = u := by
    rw [NNReal.coe_add, Real.coe_toNNReal u hu0]
    ring
  simpa only [htime] using heq.symm

/-- Joint nonvacuity of the frozen Brownian generator hypotheses,
Section 4.3: constant bounded coefficients, a genuine Brownian past
state, nonzero weight and a normalized nonzero C2 test. -/
example : Continuous (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    (∀ k : Fin 1, Continuous (fun _ : EucSpace 1 => (k : ℝ) + 1)) ∧
    (∀ x : EucSpace 1, ‖(fun _ : EucSpace 1 => (0 : EucSpace 1)) x‖ ≤ (1 : NNReal)) ∧
    (∀ k : Fin 1, |(k : ℝ) + 1| ≤ (1 : NNReal)) ∧ (1 : NNReal) ≤ 2 ∧
    StronglyMeasurable[brownianFiltration 1 1] (vectorBrownian 1 1) ∧
    StronglyMeasurable[brownianFiltration 1 1] (fun _ : BrownianSample 1 => (1 : ℝ)) ∧
    |(1 : ℝ)| ≤ 1 ∧ BoundedSmoothTest 2 (fun _ : EucSpace 1 => (1 : ℝ)) := by
  refine ⟨continuous_const, fun _ => continuous_const, fun _ => by simp,
    (fun k => by fin_cases k; norm_num), by norm_num,
    Filtration.stronglyAdapted_natural (fun t => (vectorBrownian_measurable 1 t).stronglyMeasurable) 1,
    stronglyMeasurable_const, by norm_num, contDiff_const, ?_⟩
  intro j hj y
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
