/-
# Integrable compensated observables on actual continuous path laws

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Joint measurability and uniform generator bounds justify changing the
order of time and probability integration in the martingale identity.
-/

import Transformer.BatchSize.Section4_BoundedIntervalLimits
import Transformer.BatchSize.Section4_BrownianPaths

open MeasureTheory Filter

noncomputable section

namespace Transformer.BatchSize

/-- Joint measurability of time and evaluation of any genuine
measurable continuous-path sampler, Section 4.3 (2)--(3). -/
theorem continuousPath_measurable_time_sample {Ω : Type*} [MeasurableSpace Ω] {d : ℕ}
    (Z : Ω → DiffusionPath d) (hZ : Measurable Z) :
    Measurable (fun p : ℝ × Ω => Z p.2 p.1) :=
  measurable_uncurry_of_continuous_of_measurable
    (fun ω => (Z ω).continuous) (fun u => (ContinuousMap.measurable_eval u).comp hZ)

/-- A bounded continuous generator and bounded C2 test yield
integrable weighted compensated path observables. The weighted
generator identity then gives zero mean, Section 4.3 (2)--(3).
All Fubini and Bochner integrability premises are verified explicitly. -/
theorem continuousPath_weighted_compensation {Ω : Type*} [MeasurableSpace Ω] {d : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → DiffusionPath d) (hZ : Measurable Z)
    (F : Ω → ℝ) (hFm : Measurable F) (hF1 : ∀ ω, |F ω| ≤ 1)
    (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ)
    (g : EucSpace d → ℝ) (hg : Continuous g) (K : ℝ) (hK : ∀ x, |g x| ≤ K)
    (s t : ℝ) (hst : s ≤ t)
    (hid : (∫ ω, F ω * φ (Z ω t) ∂P) - (∫ ω, F ω * φ (Z ω s) ∂P) =
      ∫ u in s..t, ∫ ω, F ω * g (Z ω u) ∂P) :
    Integrable (fun ω => F ω * (φ (Z ω t) - φ (Z ω s) - ∫ u in s..t, g (Z ω u))) P ∧
      (∫ ω, F ω * (φ (Z ω t) - φ (Z ω s) - ∫ u in s..t, g (Z ω u)) ∂P) = 0 := by
  let G (u : ℝ) (ω : Ω) := F ω * g (Z ω u)
  have hjoint := continuousPath_measurable_time_sample Z hZ
  have hGm : Measurable (Function.uncurry G) :=
    (hFm.comp measurable_snd).mul (hg.measurable.comp hjoint)
  have hGb (u : ℝ) (ω : Ω) : ‖G u ω‖ ≤ K := by
    rw [Real.norm_eq_abs]
    change |F ω * g (Z ω u)| ≤ K
    rw [abs_mul]
    exact (mul_le_mul (hF1 ω) (hK _) (abs_nonneg _) (by norm_num)).trans_eq (one_mul _)
  let : IsFiniteMeasure (volume.restrict (Set.uIoc s t)) :=
    ⟨by rw [Measure.restrict_apply_univ, Real.volume_uIoc]; exact ENNReal.ofReal_lt_top⟩
  have hGi : Integrable (Function.uncurry G) ((volume.restrict (Set.uIoc s t)).prod P) :=
    (integrable_const K).mono' hGm.stronglyMeasurable.aestronglyMeasurable
      (Eventually.of_forall fun p => hGb p.1 p.2)
  have hφi (v : ℝ) : Integrable (fun ω => F ω * φ (Z ω v)) P := by
    apply (integrable_const (1 : ℝ)).mono'
      (hFm.mul (hφ.1.continuous.measurable.comp ((ContinuousMap.measurable_eval v).comp hZ))).aestronglyMeasurable
    exact Eventually.of_forall fun ω => by
      have hφ1 : |φ (Z ω v)| ≤ 1 := by
        simpa only [norm_iteratedFDeriv_zero, Real.norm_eq_abs] using hφ.2 0 (by norm_num) (Z ω v)
      simp only [Pi.mul_apply, Function.comp_def]
      rw [Real.norm_eq_abs, abs_mul]
      exact (mul_le_mul (hF1 ω) hφ1 (abs_nonneg _) (by norm_num)).trans_eq (one_mul _)
  have hIm : Measurable (fun ω => ∫ u in s..t, G u ω) := by
    simp_rw [intervalIntegral.integral_of_le hst]
    exact (hGm.comp measurable_swap).stronglyMeasurable.integral_prod_right'.measurable
  have hIi : Integrable (fun ω => ∫ u in s..t, G u ω) P :=
    (integrable_const (K * |t - s|)).mono' hIm.aestronglyMeasurable
      (Eventually.of_forall fun ω => intervalIntegral.norm_integral_le_of_norm_le_const (fun u _ => hGb u ω))
  have heq (ω : Ω) : F ω * (φ (Z ω t) - φ (Z ω s) - ∫ u in s..t, g (Z ω u)) =
      F ω * φ (Z ω t) - F ω * φ (Z ω s) - ∫ u in s..t, G u ω := by
    rw [intervalIntegral.integral_const_mul]
    ring
  have hci := ((hφi t).sub (hφi s)).sub hIi
  have hdiff : Integrable (fun ω => F ω * φ (Z ω t) - F ω * φ (Z ω s)) P :=
    (hφi t).sub (hφi s)
  refine ⟨?_, ?_⟩
  · simp_rw [heq]
    exact hci
  simp_rw [heq]
  rw [integral_sub hdiff hIi, integral_sub (hφi t) (hφi s),
    ← intervalIntegral_integral_swap hGi, hid]
  exact sub_self _

/-- Joint nonvacuity of the compensated-path hypotheses,
Section 4.3: a nonzero constant path and test with zero continuous
generator, unit weight and a positive physical interval. -/
example : ∃ Z : Unit → DiffusionPath 1,
    Measurable Z ∧ Measurable (fun _ : Unit => (1 : ℝ)) ∧ |(1 : ℝ)| ≤ 1 ∧
    BoundedSmoothTest 2 (fun _ : EucSpace 1 => (1 : ℝ)) ∧
    Continuous (fun _ : EucSpace 1 => (0 : ℝ)) ∧ (∀ x : EucSpace 1, |(fun _ => (0 : ℝ)) x| ≤ 1) ∧
    (0 : ℝ) ≤ 1 ∧
    ((∫ ω, (1 : ℝ) * (fun _ : EucSpace 1 => (1 : ℝ)) (Z ω 1) ∂Measure.dirac ()) -
      (∫ ω, (1 : ℝ) * (fun _ : EucSpace 1 => (1 : ℝ)) (Z ω 0) ∂Measure.dirac ()) =
      ∫ u in (0 : ℝ)..1, ∫ ω, (1 : ℝ) * (fun _ : EucSpace 1 => (0 : ℝ)) (Z ω u) ∂Measure.dirac ()) := by
  refine ⟨(fun _ => ContinuousMap.const ℝ (EuclideanSpace.single (0 : Fin 1) (1 : ℝ))),
    measurable_const, measurable_const, by norm_num, ⟨contDiff_const, ?_⟩,
    continuous_const, (fun _ => by norm_num), by norm_num, by simp⟩
  intro j hj y
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
