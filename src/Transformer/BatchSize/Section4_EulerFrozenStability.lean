/-
# Stability when Euler coefficients are frozen at a lagged state

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The perturbation estimate compares a fine update with a coarse update
whose coefficients remain frozen across a refinement interval.
-/

import Transformer.BatchSize.Section4_EulerNormBounds

open MeasureTheory
open scoped NNReal BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- A Brownian Euler update whose state is Y and whose coefficients
are frozen at U, Section 4.3 (2)--(3). -/
def frozenEulerStep {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (s t : ℝ≥0) (Y U : BrownianSample d → EucSpace d)
    (ω : BrownianSample d) : EucSpace d :=
  Y ω + ((t : ℝ) - s) • b (U ω) +
    brownianKick (fun ω => WithLp.toLp 2 (a (U ω))) s t ω

/-- Mean-square stability under freezing at a lagged adapted state,
Section 4.3 (2)--(3). The forcing error is multiplied by elapsed time;
therefore a vanishing local lag error can be summed over a whole horizon. -/
theorem frozenEulerStep_stability {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (Kb Ka : ℝ≥0)
    (hb : LipschitzWith Kb b) (ha : ∀ k, LipschitzWith Ka (fun x => a x k))
    (s t : ℝ≥0) (hst : s ≤ t) (hstep : (t : ℝ) - s ≤ 1)
    (X Y U : BrownianSample d → EucSpace d)
    (hX : StronglyMeasurable[brownianFiltration d s] X)
    (hY : StronglyMeasurable[brownianFiltration d s] Y)
    (hU : StronglyMeasurable[brownianFiltration d s] U)
    (hXL2 : MemLp X 2 (brownianNoiseLaw d)) (hYL2 : MemLp Y 2 (brownianNoiseLaw d))
    (hUL2 : MemLp U 2 (brownianNoiseLaw d)) :
    (∫ ω, ‖brownianEulerStep b a s t X ω - frozenEulerStep b a s t Y U ω‖ ^ 2 ∂brownianNoiseLaw d) ≤
      (1 + (1 + 4 * Kb ^ 2 + 2 * (d : ℝ) * Ka ^ 2) * ((t : ℝ) - s)) *
        (∫ ω, ‖X ω - Y ω‖ ^ 2 ∂brownianNoiseLaw d) +
      (4 * Kb ^ 2 + 2 * (d : ℝ) * Ka ^ 2) * ((t : ℝ) - s) *
        (∫ ω, ‖Y ω - U ω‖ ^ 2 ∂brownianNoiseLaw d) := by
  let δ : ℝ := (t : ℝ) - s
  have hδ : 0 ≤ δ := sub_nonneg.mpr (NNReal.coe_le_coe.mpr hst)
  let D : BrownianSample d → EucSpace d :=
    fun ω => X ω - Y ω + δ • (b (X ω) - b (U ω))
  let H : BrownianSample d → EucSpace d :=
    fun ω => WithLp.toLp 2 (fun k => a (X ω) k - a (U ω) k)
  have hD : StronglyMeasurable[brownianFiltration d s] D :=
    (hX.sub hY).add (((hb.continuous.comp_stronglyMeasurable hX).sub
      (hb.continuous.comp_stronglyMeasurable hU)).const_smul δ)
  have hDL2 : MemLp D 2 (brownianNoiseLaw d) :=
    (hXL2.sub hYL2).add (((lipschitz_comp_memLp_finite hb hXL2).sub
      (lipschitz_comp_memLp_finite hb hUL2)).const_smul δ)
  have hH : StronglyMeasurable[brownianFiltration d s] H := by
    let : MeasurableSpace (BrownianSample d) := brownianFiltration d s
    apply (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.continuous.comp_stronglyMeasurable
    apply Measurable.stronglyMeasurable
    apply Measurable.of_eval
    intro k
    exact ((ha k).continuous.comp_stronglyMeasurable hX).measurable.sub
      ((ha k).continuous.comp_stronglyMeasurable hU).measurable
  have hHL2 : MemLp H 2 (brownianNoiseLaw d) := by
    apply MemLp.of_eval_piLp
    intro k
    exact (lipschitz_comp_memLp_finite (ha k) hXL2).sub
      (lipschitz_comp_memLp_finite (ha k) hUL2)
  have heq (ω : BrownianSample d) : brownianEulerStep b a s t X ω - frozenEulerStep b a s t Y U ω =
      D ω + brownianKick H s t ω := by
    ext k
    simp only [brownianEulerStep, frozenEulerStep, brownianKick, D, H,
      PiLp.add_apply, PiLp.sub_apply, PiLp.smul_apply, smul_eq_mul]
    ring
  simp_rw [heq]
  rw [brownianKick_energy D H s t hst hD hH hDL2 hHL2]
  let V : BrownianSample d → ℝ := fun ω => ‖X ω - Y ω‖ ^ 2 + ‖Y ω - U ω‖ ^ 2
  have hVint : Integrable V (brownianNoiseLaw d) :=
    ((hXL2.sub hYL2).integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0)).add
      ((hYL2.sub hUL2).integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0))
  have hDpoint (ω : BrownianSample d) :
      ‖D ω‖ ^ 2 ≤ (1 + δ) * ‖X ω - Y ω‖ ^ 2 + 4 * Kb ^ 2 * δ * V ω := by
    have h := norm_add_time_smul_sq_le (X ω - Y ω) (b (X ω) - b (U ω)) δ hδ
    have hb2 : ‖b (X ω) - b (U ω)‖ ^ 2 ≤ Kb ^ 2 * ‖X ω - U ω‖ ^ 2 := by
      simpa only [mul_pow] using (sq_le_sq₀ (norm_nonneg _) (by positivity)).mpr (hb.norm_sub_le _ _)
    have hl : ‖X ω - U ω‖ ^ 2 ≤ 2 * V ω := by
      dsimp [V]
      nlinarith [norm_sub_split_sq_le (X ω) (Y ω) (U ω)]
    have h1 : ‖D ω‖ ^ 2 ≤ (1 + δ) * ‖X ω - Y ω‖ ^ 2 +
        δ * (1 + δ) * (Kb ^ 2 * (2 * V ω)) := by
      exact h.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_left
        (hb2.trans (mul_le_mul_of_nonneg_left hl (sq_nonneg (Kb : ℝ)))) (by positivity)))
    have hV : 0 ≤ V ω := add_nonneg (sq_nonneg _) (sq_nonneg _)
    nlinarith [mul_nonneg (mul_nonneg (mul_nonneg (sq_nonneg (Kb : ℝ)) hδ) hV)
      (sub_nonneg.mpr hstep)]
  have hHpoint (ω : BrownianSample d) : ‖H ω‖ ^ 2 ≤ 2 * (d : ℝ) * Ka ^ 2 * V ω := by
    have hnorm : ‖H ω‖ ^ 2 ≤ (d : ℝ) * Ka ^ 2 * ‖X ω - U ω‖ ^ 2 := by
      rw [EuclideanSpace.real_norm_sq_eq]
      calc
        _ ≤ ∑ _ : Fin d, Ka ^ 2 * ‖X ω - U ω‖ ^ 2 := by
          apply Finset.sum_le_sum
          intro k hk
          have hh := (ha k).norm_sub_le (X ω) (U ω)
          rw [Real.norm_eq_abs] at hh
          simpa only [H, sq_abs, mul_pow] using (sq_le_sq₀ (abs_nonneg _) (by positivity)).mpr hh
        _ = _ := by simp [mul_assoc]
    have hs := mul_le_mul_of_nonneg_left (norm_sub_split_sq_le (X ω) (Y ω) (U ω))
      (show 0 ≤ (d : ℝ) * Ka ^ 2 by positivity)
    exact hnorm.trans (by simpa [V, mul_add, mul_assoc, mul_left_comm, mul_comm] using hs)
  have hDest := integral_mono (hDL2.integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0))
    (((hXL2.sub hYL2).integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0)).const_mul (1 + δ) |>.add
      (hVint.const_mul (4 * Kb ^ 2 * δ))) hDpoint
  have hHest := integral_mono (hHL2.integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0))
    (hVint.const_mul (2 * (d : ℝ) * Ka ^ 2)) hHpoint
  have hVeq : (∫ ω, V ω ∂brownianNoiseLaw d) =
      (∫ ω, ‖X ω - Y ω‖ ^ 2 ∂brownianNoiseLaw d) +
        (∫ ω, ‖Y ω - U ω‖ ^ 2 ∂brownianNoiseLaw d) := by
    exact integral_add ((hXL2.sub hYL2).integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0))
      ((hYL2.sub hUL2).integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0))
  simp only [Pi.add_apply] at hDest
  rw [integral_add
    (((hXL2.sub hYL2).integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0)).const_mul (1 + δ))
    (hVint.const_mul (4 * Kb ^ 2 * δ))] at hDest
  simp only [integral_const_mul, Pi.sub_apply] at hDest hHest
  calc
    _ ≤ (1 + δ) * (∫ ω, ‖X ω - Y ω‖ ^ 2 ∂brownianNoiseLaw d) +
        4 * Kb ^ 2 * δ * (∫ ω, V ω ∂brownianNoiseLaw d) +
        (2 * (d : ℝ) * Ka ^ 2 * (∫ ω, V ω ∂brownianNoiseLaw d)) * δ :=
      add_le_add hDest (mul_le_mul_of_nonneg_right hHest hδ)
    _ = _ := by rw [hVeq]; dsimp [δ]; ring

/-- Joint nonvacuity of frozen-step hypotheses, Section 4.3:
linear drift, constant amplitudes, and three equal past Brownian states. -/
example : LipschitzWith 1 (id : EucSpace 2 → EucSpace 2) ∧
    (∀ k : Fin 2, LipschitzWith 0 (fun _ : EucSpace 2 => (k : ℝ) + 1)) ∧
    (1 : ℝ≥0) ≤ 2 ∧ ((2 : ℝ) - 1 ≤ 1) ∧
    StronglyMeasurable[brownianFiltration 2 1] (vectorBrownian 2 1) ∧
    MemLp (vectorBrownian 2 1) 2 (brownianNoiseLaw 2) := by
  refine ⟨LipschitzWith.id, fun _ => LipschitzWith.const _, by norm_num, by norm_num,
    Filtration.stronglyAdapted_natural (fun t => (vectorBrownian_measurable 2 t).stronglyMeasurable) 1, ?_⟩
  apply MemLp.of_eval_piLp
  intro k
  exact ((coordinateBrownian_isBrownian k).isGaussianProcess.hasGaussianLaw_eval 1).memLp_two

end Transformer.BatchSize
