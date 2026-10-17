/-
# The uniform law on the sphere from Lebesgue measure on a shell

Lebesgue measure on the shell `1 < ‖y‖ < 2` of `ℝ^d`, projected radially onto
the sphere, is rotation invariant, finite and nonzero (`shellMeasure`).  Its
normalization `shellLaw` is a rotation-invariant probability measure, so its
`n`-fold product is a `UniformTuple` (`uniformTuple_shellLaw`): the uniform law
of `p:beta0` exists.  Being the only one (`Perspective.SphereUnique`), it lets
the null sets of the uniform law be computed with Lebesgue measure on the
product of shells (`Perspective.UniformLipschitz`), where radial projection,
`2`-Lipschitz outside the unit ball (`dist_sphereNormalize_le`), lifts Lipschitz
maps of configurations to the shells.

Source: arXiv:2312.10794v5, §4 (the uniform law `σ_d` of `p:beta0`).
-/

import Transformer.Perspective.UniformSlices
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace

open MeasureTheory Set

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- `Subtype.val` embeds the sphere, with its Borel σ-algebra, measurably into
`ℝ^d`. -/
theorem measurableEmbedding_coe_sphere :
    MeasurableEmbedding ((↑) : SSphere d → EucSpace d) :=
  Metric.isClosed_sphere.isClosedEmbedding_subtypeVal.measurableEmbedding

/-- The open shell `1 < ‖y‖ < 2` of `ℝ^d`. -/
def sphereShell (d : ℕ) : Set (EucSpace d) := {y | 1 < ‖y‖ ∧ ‖y‖ < 2}

theorem isOpen_sphereShell : IsOpen (sphereShell d) :=
  (isOpen_lt continuous_const continuous_norm).inter (isOpen_lt continuous_norm continuous_const)

/-- Linear isometries preserve the shell. -/
theorem preimage_sphereShell (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    U ⁻¹' sphereShell d = sphereShell d := by
  ext y
  simp [sphereShell]

/-- The shell avoids the origin. -/
theorem ne_zero_of_mem_sphereShell {y : EucSpace d} (hy : y ∈ sphereShell d) : y ≠ 0 :=
  norm_pos_iff.mp (by linarith [hy.1])

/-- **Radial projection onto the sphere,** `y ↦ y / ‖y‖`; the origin, outside the
shell, goes to the first basis vector. -/
noncomputable def sphereNormalize (hd : 1 ≤ d) (y : EucSpace d) : SSphere d :=
  ⟨if y = 0 then (firstBasisPoint hd : EucSpace d) else ‖y‖⁻¹ • y, by
    split_ifs with h
    · exact (firstBasisPoint hd).2
    · rw [mem_sphere_zero_iff_norm, norm_smul, norm_inv, norm_norm,
        inv_mul_cancel₀ (norm_ne_zero_iff.mpr h)]⟩

theorem coe_sphereNormalize (hd : 1 ≤ d) {y : EucSpace d} (hy : y ≠ 0) :
    (sphereNormalize hd y : EucSpace d) = ‖y‖⁻¹ • y := by
  simp [sphereNormalize, hy]

/-- The hypotheses of `coe_sphereNormalize` hold for a basis vector of `ℝ^2`. -/
example : 1 ≤ 2 ∧ (EuclideanSpace.single (0 : Fin 2) (1 : ℝ)) ≠ 0 :=
  ⟨by norm_num, by simp⟩

theorem measurable_sphereNormalize (hd : 1 ≤ d) : Measurable (sphereNormalize hd) :=
  measurableEmbedding_coe_sphere.measurable_comp_iff.mp
    (Measurable.ite (measurableSet_singleton 0) measurable_const
      (measurable_norm.inv.smul measurable_id))

/-- Radial projection commutes with linear isometries. -/
theorem sphereMap_sphereNormalize (hd : 1 ≤ d) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    {y : EucSpace d} (hy : y ≠ 0) :
    sphereMap d U (sphereNormalize hd y) = sphereNormalize hd (U y) := by
  have hUy : U y ≠ 0 := norm_ne_zero_iff.mp (by rw [U.norm_map]; exact norm_ne_zero_iff.mpr hy)
  apply Subtype.ext
  simp only [sphereMap, coe_sphereNormalize hd hy, coe_sphereNormalize hd hUy, map_smul,
    U.norm_map]

/-- The hypotheses of `sphereMap_sphereNormalize` hold for the identity of `ℝ^2`
and a basis vector. -/
example : 1 ≤ 2 ∧ (EuclideanSpace.single (0 : Fin 2) (1 : ℝ)) ≠ 0 :=
  ⟨by norm_num, by simp⟩

/-- **Radial projection is `2`-Lipschitz outside the unit ball.** -/
theorem dist_sphereNormalize_le (hd : 1 ≤ d) {y y' : EucSpace d} (hy : 1 ≤ ‖y‖)
    (hy' : 1 ≤ ‖y'‖) : dist (sphereNormalize hd y) (sphereNormalize hd y') ≤ 2 * dist y y' := by
  have ha : 0 < ‖y‖ := by linarith
  have hb : 0 < ‖y'‖ := by linarith
  rw [Subtype.dist_eq, coe_sphereNormalize hd (norm_pos_iff.mp ha),
    coe_sphereNormalize hd (norm_pos_iff.mp hb), dist_eq_norm, dist_eq_norm]
  have hsplit : ‖y‖⁻¹ • y - ‖y'‖⁻¹ • y' = ‖y‖⁻¹ • (y - y') + (‖y‖⁻¹ - ‖y'‖⁻¹) • y' := by
    rw [smul_sub, sub_smul]
    abel
  have h1 : ‖‖y‖⁻¹ • (y - y')‖ ≤ ‖y - y'‖ := by
    rw [norm_smul, norm_inv, norm_norm]
    exact mul_le_of_le_one_left (norm_nonneg _) (inv_le_one_of_one_le₀ hy)
  have h2 : ‖(‖y‖⁻¹ - ‖y'‖⁻¹) • y'‖ ≤ ‖y - y'‖ := by
    have e : (‖y‖⁻¹ - ‖y'‖⁻¹) * ‖y'‖ = (‖y'‖ - ‖y‖) / ‖y‖ := by
      field_simp
    calc ‖(‖y‖⁻¹ - ‖y'‖⁻¹) • y'‖ = |(‖y‖⁻¹ - ‖y'‖⁻¹) * ‖y'‖| := by
          rw [norm_smul, Real.norm_eq_abs, abs_mul, abs_norm]
      _ = |‖y'‖ - ‖y‖| / ‖y‖ := by rw [e, abs_div, abs_of_pos ha]
      _ ≤ |‖y'‖ - ‖y‖| := div_le_self (abs_nonneg _) hy
      _ ≤ ‖y' - y‖ := abs_norm_sub_norm_le _ _
      _ = ‖y - y'‖ := norm_sub_rev _ _
  rw [hsplit]
  linarith [norm_add_le (‖y‖⁻¹ • (y - y')) ((‖y‖⁻¹ - ‖y'‖⁻¹) • y')]

/-- The hypotheses of `dist_sphereNormalize_le` hold for a basis vector of `ℝ²`. -/
example : 1 ≤ ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ := by simp

/-- **The shell measure:** Lebesgue measure on the shell `1 < ‖y‖ < 2`, projected
radially onto the sphere. -/
noncomputable def shellMeasure (hd : 1 ≤ d) : Measure (SSphere d) :=
  (volume.restrict (sphereShell d)).map (sphereNormalize hd)

/-- **The shell measure is rotation invariant:** Lebesgue measure and the shell
are, and radial projection commutes with rotations. -/
theorem shellMeasure_isometry (hd : 1 ≤ d) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    (shellMeasure hd).map (sphereMap d U) = shellMeasure hd := by
  have hS : MeasurableSet (sphereShell d) := isOpen_sphereShell.measurableSet
  have hU : MeasurePreserving U (volume.restrict (sphereShell d))
      (volume.restrict (sphereShell d)) := by
    have h := U.measurePreserving.restrict_preimage hS
    rwa [preimage_sphereShell] at h
  rw [shellMeasure, Measure.map_map (measurable_sphereMap d U) (measurable_sphereNormalize hd)]
  conv_rhs => rw [← hU.map_eq, Measure.map_map (measurable_sphereNormalize hd)
    U.continuous.measurable]
  refine Measure.map_congr ?_
  filter_upwards [ae_restrict_mem hS] with y hy
  exact sphereMap_sphereNormalize hd U (ne_zero_of_mem_sphereShell hy)

theorem shellMeasure_univ (hd : 1 ≤ d) : shellMeasure hd univ = volume (sphereShell d) := by
  rw [shellMeasure, Measure.map_apply (measurable_sphereNormalize hd) MeasurableSet.univ,
    preimage_univ, Measure.restrict_apply_univ]

theorem volume_sphereShell_pos (hd : 1 ≤ d) : 0 < volume (sphereShell d) := by
  refine isOpen_sphereShell.measure_pos volume
    ⟨(3 / 2 : ℝ) • (firstBasisPoint hd : EucSpace d), ?_⟩
  have h1 : ‖(firstBasisPoint hd : EucSpace d)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (firstBasisPoint hd).2
  simp only [sphereShell, mem_ofPred_eq, norm_smul, h1, Real.norm_eq_abs]
  norm_num

theorem volume_sphereShell_lt_top : volume (sphereShell d) < ⊤ :=
  (measure_mono fun _ hy => mem_ball_zero_iff.mpr hy.2).trans_lt measure_ball_lt_top

instance (hd : 1 ≤ d) : IsFiniteMeasure (shellMeasure hd) :=
  ⟨by rw [shellMeasure_univ]; exact volume_sphereShell_lt_top⟩

/-- **The shell law:** the shell measure normalized to unit mass. -/
noncomputable def shellLaw (hd : 1 ≤ d) : Measure (SSphere d) :=
  (volume (sphereShell d))⁻¹ • shellMeasure hd

instance isProbabilityMeasure_shellLaw (hd : 1 ≤ d) : IsProbabilityMeasure (shellLaw hd) :=
  ⟨by rw [shellLaw, Measure.smul_apply, shellMeasure_univ, smul_eq_mul,
    ENNReal.inv_mul_cancel (volume_sphereShell_pos hd).ne' volume_sphereShell_lt_top.ne]⟩

/-- The shell law is rotation invariant. -/
theorem shellLaw_isometry (hd : 1 ≤ d) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    (shellLaw hd).map (sphereMap d U) = shellLaw hd := by
  rw [shellLaw, Measure.map_smul _ (measurable_sphereMap d U).aemeasurable, shellMeasure_isometry]

/-- **The uniform law exists:** the product of shell laws is a `UniformTuple`. -/
theorem uniformTuple_shellLaw (hd : 1 ≤ d) :
    UniformTuple d n (Measure.pi fun _ : Idx n => shellLaw hd) :=
  ⟨shellLaw hd, isProbabilityMeasure_shellLaw hd, shellLaw_isometry hd, rfl⟩

/-- The hypothesis of the `shellLaw` lemmas holds for the circle, `d = 2`. -/
example : 1 ≤ 2 := by norm_num

end Perspective
end Transformer
