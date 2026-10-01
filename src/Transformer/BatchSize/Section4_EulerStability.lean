/-
# Mean-square stability of the actual Brownian Euler update

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Two adapted random states are coupled using the same fresh Brownian
increment. Exact adapted energy removes the drift-noise cross term.
-/

import Transformer.BatchSize.Section4_BrownianKick

open MeasureTheory
open scoped NNReal BigOperators

noncomputable section

namespace Transformer.BatchSize

/-- One actual Brownian Euler update with a random current state,
Section 4.3 (2)--(3). -/
def brownianEulerStep {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (s t : ℝ≥0) (X : BrownianSample d → EucSpace d)
    (ω : BrownianSample d) : EucSpace d :=
  X ω + ((t : ℝ) - s) • b (X ω) +
    brownianKick (fun ω => WithLp.toLp 2 (a (X ω))) s t ω

/-- Mean-square stability of a coupled Euler step, Section 4.3
(2)--(3). The coefficient bound uses the scalar coordinate Lipschitz
constant, and therefore retains the dimension factor in the noise term. -/
theorem brownianEulerStep_stability {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (Kb Ka : ℝ≥0)
    (hb : LipschitzWith Kb b) (ha : ∀ k, LipschitzWith Ka (fun x => a x k))
    (s t : ℝ≥0) (hst : s ≤ t) (X Y : BrownianSample d → EucSpace d)
    (hX : StronglyMeasurable[brownianFiltration d s] X)
    (hY : StronglyMeasurable[brownianFiltration d s] Y)
    (hXL2 : MemLp X 2 (brownianNoiseLaw d)) (hYL2 : MemLp Y 2 (brownianNoiseLaw d)) :
    (∫ ω, ‖brownianEulerStep b a s t X ω - brownianEulerStep b a s t Y ω‖ ^ 2 ∂brownianNoiseLaw d) ≤
      ((1 + ((t : ℝ) - s) * Kb) ^ 2 + (d : ℝ) * Ka ^ 2 * ((t : ℝ) - s)) *
        (∫ ω, ‖X ω - Y ω‖ ^ 2 ∂brownianNoiseLaw d) := by
  let δ : ℝ := (t : ℝ) - s
  have hδ : 0 ≤ δ := sub_nonneg.mpr (NNReal.coe_le_coe.mpr hst)
  let D : BrownianSample d → EucSpace d :=
    fun ω => X ω - Y ω + δ • (b (X ω) - b (Y ω))
  let H : BrownianSample d → EucSpace d :=
    fun ω => WithLp.toLp 2 (fun k => a (X ω) k - a (Y ω) k)
  have hD : StronglyMeasurable[brownianFiltration d s] D :=
    (hX.sub hY).add (((hb.continuous.comp_stronglyMeasurable hX).sub
      (hb.continuous.comp_stronglyMeasurable hY)).const_smul δ)
  have hDL2 : MemLp D 2 (brownianNoiseLaw d) :=
    (hXL2.sub hYL2).add (((lipschitz_comp_memLp_finite hb hXL2).sub
      (lipschitz_comp_memLp_finite hb hYL2)).const_smul δ)
  have hH : StronglyMeasurable[brownianFiltration d s] H := by
    let : MeasurableSpace (BrownianSample d) := brownianFiltration d s
    apply (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.continuous.comp_stronglyMeasurable
    apply Measurable.stronglyMeasurable
    apply Measurable.of_eval
    intro k
    exact ((ha k).continuous.comp_stronglyMeasurable hX).measurable.sub
      ((ha k).continuous.comp_stronglyMeasurable hY).measurable
  have hHL2 : MemLp H 2 (brownianNoiseLaw d) := by
    apply MemLp.of_eval_piLp
    intro k
    exact (lipschitz_comp_memLp_finite (ha k) hXL2).sub
      (lipschitz_comp_memLp_finite (ha k) hYL2)
  have heq (ω : BrownianSample d) : brownianEulerStep b a s t X ω - brownianEulerStep b a s t Y ω =
      D ω + brownianKick H s t ω := by
    ext k
    simp only [brownianEulerStep, brownianKick, D, H, PiLp.add_apply, PiLp.sub_apply,
      PiLp.smul_apply, smul_eq_mul]
    ring
  simp_rw [heq]
  rw [brownianKick_energy D H s t hst hD hH hDL2 hHL2]
  have hbase := (hXL2.sub hYL2).integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0)
  have hDpoint (ω : BrownianSample d) :
      ‖D ω‖ ^ 2 ≤ (1 + δ * Kb) ^ 2 * ‖X ω - Y ω‖ ^ 2 := by
    have hh : ‖D ω‖ ≤ (1 + δ * Kb) * ‖X ω - Y ω‖ := by
      calc
        _ ≤ ‖X ω - Y ω‖ + ‖δ • (b (X ω) - b (Y ω))‖ := norm_add_le _ _
        _ = ‖X ω - Y ω‖ + δ * ‖b (X ω) - b (Y ω)‖ := by
          rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hδ]
        _ ≤ ‖X ω - Y ω‖ + δ * (Kb * ‖X ω - Y ω‖) :=
          add_le_add le_rfl (mul_le_mul_of_nonneg_left (hb.norm_sub_le (X ω) (Y ω)) hδ)
        _ = _ := by ring
    have h := (sq_le_sq₀ (norm_nonneg _) (by positivity)).mpr hh
    simpa only [mul_pow] using h
  have hHpoint (ω : BrownianSample d) : ‖H ω‖ ^ 2 ≤ (d : ℝ) * Ka ^ 2 * ‖X ω - Y ω‖ ^ 2 := by
    rw [EuclideanSpace.real_norm_sq_eq]
    calc
      _ ≤ ∑ _ : Fin d, Ka ^ 2 * ‖X ω - Y ω‖ ^ 2 := by
        apply Finset.sum_le_sum
        intro k hk
        have hh := (ha k).norm_sub_le (X ω) (Y ω)
        rw [Real.norm_eq_abs] at hh
        have hs := (sq_le_sq₀ (abs_nonneg _) (by positivity)).mpr hh
        simpa only [H, WithLp.ofLp_toLp, sq_abs, mul_pow] using hs
      _ = _ := by simp [mul_assoc]
  have hDest : (∫ ω, ‖D ω‖ ^ 2 ∂brownianNoiseLaw d) ≤
      (1 + δ * Kb) ^ 2 * (∫ ω, ‖X ω - Y ω‖ ^ 2 ∂brownianNoiseLaw d) := by
    have h := integral_mono (hDL2.integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0))
      (hbase.const_mul ((1 + δ * Kb) ^ 2)) hDpoint
    simpa only [integral_const_mul, Pi.sub_apply] using h
  have hHest : (∫ ω, ‖H ω‖ ^ 2 ∂brownianNoiseLaw d) ≤
      ((d : ℝ) * Ka ^ 2) * (∫ ω, ‖X ω - Y ω‖ ^ 2 ∂brownianNoiseLaw d) := by
    have h := integral_mono (hHL2.integrable_norm_pow (by norm_num : (2 : ℕ) ≠ 0))
      (hbase.const_mul ((d : ℝ) * Ka ^ 2)) hHpoint
    simpa only [integral_const_mul, Pi.sub_apply] using h
  calc
    _ ≤ (1 + δ * Kb) ^ 2 * (∫ ω, ‖X ω - Y ω‖ ^ 2 ∂brownianNoiseLaw d) +
        (((d : ℝ) * Ka ^ 2) * (∫ ω, ‖X ω - Y ω‖ ^ 2 ∂brownianNoiseLaw d)) * δ :=
      add_le_add hDest (mul_le_mul_of_nonneg_right hHest hδ)
    _ = _ := by dsimp [δ]; ring

/-- Joint nonvacuity of stability hypotheses, Section 4.3:
two-dimensional linear drift, constant amplitudes, and two adapted states. -/
example : LipschitzWith 1 (id : EucSpace 2 → EucSpace 2) ∧
    (∀ k : Fin 2, LipschitzWith 0 (fun _ : EucSpace 2 => (k : ℝ) + 1)) ∧
    (1 : ℝ≥0) ≤ 2 ∧
    StronglyMeasurable[brownianFiltration 2 1] (vectorBrownian 2 1) ∧
    MemLp (vectorBrownian 2 1) 2 (brownianNoiseLaw 2) := by
  refine ⟨LipschitzWith.id, fun _ => LipschitzWith.const _, by norm_num,
    Filtration.stronglyAdapted_natural
      (fun t => (vectorBrownian_measurable 2 t).stronglyMeasurable) 1, ?_⟩
  apply MemLp.of_eval_piLp
  intro k
  exact ((coordinateBrownian_isBrownian k).isGaussianProcess.hasGaussianLaw_eval 1).memLp_two

end Transformer.BatchSize
