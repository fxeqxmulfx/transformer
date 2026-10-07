/-
# The full characteristic flow of a stationary Dirac law

ArXiv:2410.06833v1, §5, `eq: mean.field.pde` and `eq: flow.map`.
The kernel in the numerator and partition function cancel at `δ_w`.
The mass itself is stationary, but trajectories from other points of the
sphere move with velocity `Proj_z w`. The identity map is therefore not
its flow in dimensions above one.

Here the scalar coefficients give an explicit global flow, with its unit
norm, initial value, continuity and differential equation all proved.
This construction supplies the complete flow hypothesis in the
counterexample to `eq: v.small`; no flow existence claim is assumed.
-/

import Transformer.Metastability.Section5_DiracCoefficients
import Transformer.Metastability.MeanFieldStatic

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Metastability

open Perspective

/-- Explicit ambient trajectory of `eq: flow.map` at a stationary Dirac law.

Source: arXiv:2410.06833v1, §5. -/
noncomputable def diracAmbientFlow (d : ℕ) (t : ℝ) (w z : EucSpace d) : EucSpace d :=
  let c := inner (𝕜 := ℝ) z w
  diracAlong t c • w + diracAcross t c • (z - c • w)

/-- Unit initial conditions remain on the sphere in `eq: flow.map`.

Source: arXiv:2410.06833v1, §5. -/
theorem norm_diracAmbientFlow (d : ℕ) (t : ℝ) (w z : EucSpace d)
    (hw : ‖w‖ = 1) (hz : ‖z‖ = 1) : ‖diracAmbientFlow d t w z‖ = 1 := by
  let c := inner (𝕜 := ℝ) z w
  have hc : c ∈ Set.Icc (-1 : ℝ) 1 := real_inner_mem_Icc_of_norm_eq_one hz hw
  have hp : inner (𝕜 := ℝ) w (z - c • w) = 0 := by
    rw [inner_sub_right, real_inner_smul_right, real_inner_self_eq_norm_mul_norm, hw,
      real_inner_comm z w]
    dsimp [c]
    ring
  have hp2 : ‖z - c • w‖ ^ 2 = 1 - c ^ 2 := by
    rw [norm_sub_sq_real, real_inner_smul_right, hz, norm_smul, hw, mul_one,
      Real.norm_eq_abs, sq_abs]
    dsimp [c]
    ring
  have hsq : ‖diracAmbientFlow d t w z‖ ^ 2 = 1 := by
    change ‖diracAlong t c • w + diracAcross t c • (z - c • w)‖ ^ 2 = 1
    rw [norm_add_sq_real, real_inner_smul_left, real_inner_smul_right, hp, norm_smul,
      norm_smul, hw, Real.norm_eq_abs, Real.norm_eq_abs]
    simp only [mul_zero, add_zero, mul_one, mul_pow, sq_abs, hp2]
    exact dirac_coefficients_unit t c hc
  nlinarith [norm_nonneg (diracAmbientFlow d t w z)]

/-- The explicit characteristic flow of `eq: flow.map`, bundled on the unit sphere.

Source: arXiv:2410.06833v1, §5. -/
noncomputable def diracFlow (d : ℕ) (w : SSphere d) (t : ℝ) (z : SSphere d) : SSphere d :=
  ⟨diracAmbientFlow d t w z, mem_sphere_zero_iff_norm.mpr
    (norm_diracAmbientFlow d t w z (mem_sphere_zero_iff_norm.mp w.2)
      (mem_sphere_zero_iff_norm.mp z.2))⟩

/-- The constructed flow has the initial condition of `eq: flow.map`.

Source: arXiv:2410.06833v1, §5. -/
theorem diracFlow_zero (d : ℕ) (w z : SSphere d) : diracFlow d w 0 z = z := by
  apply Subtype.ext
  change diracAmbientFlow d 0 w z = z
  simp only [diracAmbientFlow, diracAlong_zero, diracAcross_zero, one_smul]
  module

/-- The longitudinal coordinate of the characteristic flow in `eq: flow.map`.

Source: arXiv:2410.06833v1, §5. -/
theorem inner_diracFlow (d : ℕ) (w z : SSphere d) (t : ℝ) :
    inner (𝕜 := ℝ) (diracFlow d w t z : EucSpace d) (w : EucSpace d) =
      diracAlong t (inner (𝕜 := ℝ) (z : EucSpace d) (w : EucSpace d)) := by
  change inner (𝕜 := ℝ) (diracAmbientFlow d t w z) w = _
  simp only [diracAmbientFlow, inner_add_left, real_inner_smul_left, inner_sub_left,
    real_inner_self_eq_norm_mul_norm, mem_sphere_zero_iff_norm.mp w.2]
  ring

/-- The mass at the center stays fixed in the characteristic flow of `eq: flow.map`.

Source: arXiv:2410.06833v1, §5. -/
theorem diracFlow_self (d : ℕ) (w : SSphere d) (t : ℝ) : diracFlow d w t w = w := by
  apply Subtype.ext
  change diracAmbientFlow d t w w = w
  simp [diracAmbientFlow, mem_sphere_zero_iff_norm.mp w.2, diracAlong_one]

/-- Each explicit trajectory solves the full normalized equation of `eq: flow.map`.

Source: arXiv:2410.06833v1, §5. -/
theorem hasDerivAt_diracFlow (d : ℕ) (β : ℝ) (w z : SSphere d) (t : ℝ) :
    HasDerivAt (fun s => (diracFlow d w s z : EucSpace d))
      (MFVel d β (diracProb d w) (diracFlow d w t z)) t := by
  let c := inner (𝕜 := ℝ) (z : EucSpace d) (w : EucSpace d)
  have hc : c ∈ Set.Icc (-1 : ℝ) 1 := real_inner_mem_Icc_of_norm_eq_one
    (mem_sphere_zero_iff_norm.mp z.2) (mem_sphere_zero_iff_norm.mp w.2)
  have h := ((hasDerivAt_diracAlong t c hc).smul_const (w : EucSpace d)).add
    ((hasDerivAt_diracAcross t c hc).smul_const ((z : EucSpace d) - c • (w : EucSpace d)))
  rw [MFVel_diracProb, proj, inner_diracFlow]
  convert h using 1
  · rfl
  · change (w : EucSpace d) - diracAlong t c • diracAmbientFlow d t w z = _
    dsimp [diracAmbientFlow, c]
    module

/-- The characteristic map in `eq: flow.map` is continuous on the entire sphere.

Source: arXiv:2410.06833v1, §5. -/
theorem continuous_diracFlow (d : ℕ) (w : SSphere d) (t : ℝ) :
    Continuous (diracFlow d w t) := by
  have hc : Continuous (fun z : SSphere d =>
      inner (𝕜 := ℝ) (z : EucSpace d) (w : EucSpace d)) :=
    continuous_subtype_val.inner continuous_const
  have hd (z : SSphere d) : diracDen t
      (inner (𝕜 := ℝ) (z : EucSpace d) (w : EucSpace d)) ≠ 0 :=
    ne_of_gt (diracDen_pos t _ (real_inner_mem_Icc_of_norm_eq_one
      (mem_sphere_zero_iff_norm.mp z.2) (mem_sphere_zero_iff_norm.mp w.2)))
  have hC : Continuous (fun z : SSphere d => diracAlong t
      (inner (𝕜 := ℝ) (z : EucSpace d) (w : EucSpace d))) := by
    unfold diracAlong diracDen at *
    apply Continuous.div
    · fun_prop
    · fun_prop
    · exact hd
  have hS : Continuous (fun z : SSphere d => diracAcross t
      (inner (𝕜 := ℝ) (z : EucSpace d) (w : EucSpace d))) := by
    unfold diracAcross diracDen at *
    exact continuous_const.div (by fun_prop) hd
  apply Continuous.subtype_mk
  exact (hC.smul continuous_const).add
    (hS.smul (continuous_subtype_val.sub (hc.smul continuous_const)))

/-- All characteristic-flow conditions of `eq: flow.map` hold for the explicit Dirac flow.

Source: arXiv:2410.06833v1, §5. -/
theorem isMFFlow_diracFlow (d : ℕ) (β : ℝ) (w : SSphere d) :
    IsMFFlow d β (fun _ => diracProb d w) (diracFlow d w) :=
  ⟨fun t => (continuous_diracFlow d w t).measurable, diracFlow_zero d w,
    fun z t => hasDerivAt_diracFlow d β w z t⟩

/-- Both unit-norm hypotheses are attained by a standard basis vector. -/
example : let e := EuclideanSpace.single (0 : Fin 1) (1 : ℝ)
    ‖e‖ = 1 ∧ ‖e‖ = 1 ∧ ‖diracAmbientFlow 1 0 e e‖ = 1 := by
  intro e
  have he : ‖e‖ = 1 := by simp [e, PiLp.norm_single]
  exact ⟨he, he, norm_diracAmbientFlow 1 0 e e he he⟩

/-- The PDE and full-flow predicates have simultaneous inhabitants. -/
example : meanFieldPDE 1 1000 (fun _ => diracProb 1 (basePoint 0)) ∧
    IsMFFlow 1 1000 (fun _ => diracProb 1 (basePoint 0)) (diracFlow 1 (basePoint 0)) :=
  ⟨meanFieldPDE_dirac 1 1000 _, isMFFlow_diracFlow 1 1000 _⟩

end Transformer.Metastability
