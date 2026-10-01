/-
# Adaptedness and local displacement of Brownian Euler steps

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The displacement bound controls the error from freezing coefficients
inside a coarse grid interval.
-/

import Transformer.BatchSize.Section4_EulerStability

open MeasureTheory
open scoped NNReal BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- A left-adapted diagonal kick is measurable at its right endpoint,
Section 4.3 (2)--(3). -/
theorem brownianKick_adapted {d : ℕ} (H : BrownianSample d → EucSpace d)
    (s t : ℝ≥0) (hst : s ≤ t)
    (hH : StronglyMeasurable[brownianFiltration d s] H) :
    StronglyMeasurable[brownianFiltration d t] (brownianKick H s t) := by
  have hst' := (brownianFiltration d).mono hst
  let : MeasurableSpace (BrownianSample d) := brownianFiltration d t
  apply (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.continuous.comp_stronglyMeasurable
  apply Measurable.stronglyMeasurable
  apply Measurable.of_eval
  intro k
  exact (((PiLp.continuous_apply 2 (fun _ : Fin d => ℝ) k).comp_stronglyMeasurable
    (hH.mono hst')).mul (((coordinateBrownian_filtered k).stronglyAdapted t).sub
      (((coordinateBrownian_filtered k).stronglyAdapted s).mono hst'))).measurable

/-- A Brownian Euler update remains adapted to the joint driver,
Section 4.3 (2)--(3). -/
theorem brownianEulerStep_adapted {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (hb : Continuous b)
    (ha : ∀ k, Continuous (fun x => a x k)) (s t : ℝ≥0) (hst : s ≤ t)
    (X : BrownianSample d → EucSpace d)
    (hX : StronglyMeasurable[brownianFiltration d s] X) :
    StronglyMeasurable[brownianFiltration d t] (brownianEulerStep b a s t X) := by
  have hA : StronglyMeasurable[brownianFiltration d s]
      (fun ω => WithLp.toLp 2 (a (X ω))) := by
    let : MeasurableSpace (BrownianSample d) := brownianFiltration d s
    apply (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.continuous.comp_stronglyMeasurable
    apply Measurable.stronglyMeasurable
    exact Measurable.of_eval fun k => ((ha k).comp_stronglyMeasurable hX).measurable
  have hD := (hX.add ((hb.comp_stronglyMeasurable hX).const_smul ((t : ℝ) - s))).mono
    ((brownianFiltration d).mono hst)
  exact hD.add (brownianKick_adapted _ s t hst hA)

/-- Global Lipschitz coefficients preserve L2 in a Brownian Euler step,
Section 4.3 (2)--(3). -/
theorem brownianEulerStep_memLp {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (Kb Ka : ℝ≥0)
    (hb : LipschitzWith Kb b) (ha : ∀ k, LipschitzWith Ka (fun x => a x k))
    (s t : ℝ≥0) (hst : s ≤ t) (X : BrownianSample d → EucSpace d)
    (hX : StronglyMeasurable[brownianFiltration d s] X)
    (hXL2 : MemLp X 2 (brownianNoiseLaw d)) :
    MemLp (brownianEulerStep b a s t X) 2 (brownianNoiseLaw d) := by
  have hA : StronglyMeasurable[brownianFiltration d s]
      (fun ω => WithLp.toLp 2 (a (X ω))) := by
    let : MeasurableSpace (BrownianSample d) := brownianFiltration d s
    apply (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.continuous.comp_stronglyMeasurable
    apply Measurable.stronglyMeasurable
    exact Measurable.of_eval fun k => ((ha k).continuous.comp_stronglyMeasurable hX).measurable
  have hAL2 : MemLp (fun ω => WithLp.toLp 2 (a (X ω))) 2 (brownianNoiseLaw d) :=
    MemLp.of_eval_piLp fun k => lipschitz_comp_memLp_finite (ha k) hXL2
  exact (hXL2.add ((lipschitz_comp_memLp_finite hb hXL2).const_smul ((t : ℝ) - s))).add
    (brownianKick_memLp _ s t hst hA hAL2)

/-- With bounded coefficients, the mean-square local lag is at most
M^2 times elapsed time squared plus d A^2 times elapsed time,
Section 4.3 (2)--(3). In the optimizer SDE, A itself scales as sqrt(eta). -/
theorem brownianEulerStep_displacement_bound {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (Kb Ka M A : ℝ≥0)
    (hb : LipschitzWith Kb b) (ha : ∀ k, LipschitzWith Ka (fun x => a x k))
    (hbM : ∀ x, ‖b x‖ ≤ M) (haA : ∀ x k, |a x k| ≤ A)
    (s t : ℝ≥0) (hst : s ≤ t) (X : BrownianSample d → EucSpace d)
    (hX : StronglyMeasurable[brownianFiltration d s] X)
    (hXL2 : MemLp X 2 (brownianNoiseLaw d)) :
    (∫ ω, ‖brownianEulerStep b a s t X ω - X ω‖ ^ 2 ∂brownianNoiseLaw d) ≤
      M ^ 2 * ((t : ℝ) - s) ^ 2 + (d : ℝ) * A ^ 2 * ((t : ℝ) - s) := by
  let δ : ℝ := (t : ℝ) - s
  let D : BrownianSample d → EucSpace d := fun ω => δ • b (X ω)
  let H : BrownianSample d → EucSpace d := fun ω => WithLp.toLp 2 (a (X ω))
  have hD : StronglyMeasurable[brownianFiltration d s] D :=
    (hb.continuous.comp_stronglyMeasurable hX).const_smul δ
  have hDL2 : MemLp D 2 (brownianNoiseLaw d) :=
    (lipschitz_comp_memLp_finite hb hXL2).const_smul δ
  have hH : StronglyMeasurable[brownianFiltration d s] H := by
    let : MeasurableSpace (BrownianSample d) := brownianFiltration d s
    apply (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.continuous.comp_stronglyMeasurable
    apply Measurable.stronglyMeasurable
    exact Measurable.of_eval fun k => ((ha k).continuous.comp_stronglyMeasurable hX).measurable
  have hHL2 : MemLp H 2 (brownianNoiseLaw d) :=
    MemLp.of_eval_piLp fun k => lipschitz_comp_memLp_finite (ha k) hXL2
  have heq (ω : BrownianSample d) : brownianEulerStep b a s t X ω - X ω =
      D ω + brownianKick H s t ω := by
    simp only [brownianEulerStep, D, H]
    abel
  simp_rw [heq]
  rw [brownianKick_energy D H s t hst hD hH hDL2 hHL2]
  have hDb : (∫ ω, ‖D ω‖ ^ 2 ∂brownianNoiseLaw d) ≤ δ ^ 2 * M ^ 2 := by
    have h := integral_mono (hDL2.integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0))
      (integrable_const (δ ^ 2 * (M : ℝ) ^ 2)) fun ω => by
        simp only [D, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs]
        exact mul_le_mul_of_nonneg_left ((sq_le_sq₀ (norm_nonneg _) M.coe_nonneg).mpr
          (hbM (X ω))) (sq_nonneg δ)
    simpa using h
  have hHb : (∫ ω, ‖H ω‖ ^ 2 ∂brownianNoiseLaw d) ≤ (d : ℝ) * A ^ 2 := by
    have hpoint (ω : BrownianSample d) : ‖H ω‖ ^ 2 ≤ (d : ℝ) * A ^ 2 := by
      rw [EuclideanSpace.real_norm_sq_eq]
      calc
        _ ≤ ∑ _ : Fin d, (A : ℝ) ^ 2 := by
          apply Finset.sum_le_sum
          intro k hk
          simpa only [H, sq_abs] using
            (sq_le_sq₀ (abs_nonneg _) A.coe_nonneg).mpr (haA (X ω) k)
        _ = _ := by simp
    have h := integral_mono (hHL2.integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0))
      (integrable_const ((d : ℝ) * A ^ 2)) hpoint
    simpa using h
  calc
    _ ≤ δ ^ 2 * M ^ 2 + ((d : ℝ) * A ^ 2) * δ :=
      add_le_add hDb (mul_le_mul_of_nonneg_right hHb
        (sub_nonneg.mpr (NNReal.coe_le_coe.mpr hst)))
    _ = _ := by dsimp [δ]; ring

/-- Joint nonvacuity of step bounds, Section 4.3: zero drift,
positive constant amplitude, and an actual past Brownian state. -/
example : LipschitzWith 0 (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    (∀ _ : Fin 1, LipschitzWith 0 (fun _ : EucSpace 1 => (1 : ℝ))) ∧
    (∀ x : EucSpace 1, ‖(fun _ : EucSpace 1 => (0 : EucSpace 1)) x‖ ≤ (1 : ℝ≥0)) ∧
    (∀ (x : EucSpace 1) (k : Fin 1), |(fun _ : EucSpace 1 => fun _ : Fin 1 => (1 : ℝ)) x k| ≤
      (1 : ℝ≥0)) ∧ (1 : ℝ≥0) ≤ 2 ∧
    StronglyMeasurable[brownianFiltration 1 1] (vectorBrownian 1 1) ∧
    MemLp (vectorBrownian 1 1) 2 (brownianNoiseLaw 1) := by
  refine ⟨LipschitzWith.const _, fun _ => LipschitzWith.const _, by simp, by simp,
    by norm_num, Filtration.stronglyAdapted_natural
      (fun t => (vectorBrownian_measurable 1 t).stronglyMeasurable) 1, ?_⟩
  apply MemLp.of_eval_piLp
  intro k
  exact ((coordinateBrownian_isBrownian k).isGaussianProcess.hasGaussianLaw_eval 1).memLp_two

end Transformer.BatchSize
