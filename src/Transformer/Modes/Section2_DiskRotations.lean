/-
# Rotations preserve the actual oscillating-field law

Source: arXiv:2412.09080v3, §2.1, the joint-law hypothesis
in `thm:kac-rice`. The signed scale is uniform on `(-1,1)`
and the disk coordinate is independently uniform in the unit
disk. Multiplication by the actual oscillation phase is a
real linear isometry, so it preserves disk volume and the
normalized disk probability law.

The rotation angle depends on the scale. Fubini proves that
the resulting measurable map of the product probability space
still preserves its actual law. The joint observation of the
field and its actual derivative at every fixed time therefore
has the same law as the explicit time-zero map
`(r,z) ↦ (r*(2+re z), -im z)`.

The actual derivative differs from that expression at scale
zero; the proved nullity of that scale justifies equality of
the laws. No joint density is presumed in this step.
-/

import Transformer.Modes.Section2_DiskProbability
import Transformer.Modes.Section2_OscillatingModulus
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace

open Real Set Filter MeasureTheory

namespace Transformer.Modes

/-- Multiplication by the actual unit phase is an explicit real linear isometry of the complex disk coordinates.
Source: arXiv:2412.09080v3, §2.1, the disk rotation in the field testing `thm:kac-rice`. -/
noncomputable def riceCounterRotation (r t : ℝ) : ℂ ≃ₗᵢ[ℝ] ℂ where
  toFun := fun z => z * riceCounterPhase r t
  invFun := fun z => z / riceCounterPhase r t
  left_inv := fun z => mul_div_cancel_right₀ z (Complex.exp_ne_zero _)
  right_inv := fun z => div_mul_cancel₀ z (Complex.exp_ne_zero _)
  map_add' := fun z w => add_mul z w _
  map_smul' := fun a z => smul_mul_assoc a z _
  norm_map' := fun z => norm_mul_riceCounterPhase z r t

/-- The actual disk isometry is exactly multiplication by the original oscillation phase.
Source: arXiv:2412.09080v3, §2.1, the phase coordinates used for the joint-law hypothesis in `thm:kac-rice`. -/
theorem riceCounterRotation_apply (r t : ℝ) (z : ℂ) :
    riceCounterRotation r t z = z * riceCounterPhase r t := rfl

/-- The actual phase rotation preserves the original normalized disk probability law.
Source: arXiv:2412.09080v3, §2.1, rotation invariance of the disk parameter used for the joint law in `thm:kac-rice`. -/
theorem measurePreserving_riceCounterRotation (r t : ℝ) :
    MeasurePreserving (riceCounterRotation r t) riceCounterDiskLaw riceCounterDiskLaw := by
  have h := (riceCounterRotation r t).measurePreserving
  have hpre : (riceCounterRotation r t) ⁻¹' Metric.ball (0 : ℂ) 1 = Metric.ball 0 1 := by
    ext z
    simp only [mem_preimage, Metric.mem_ball, dist_zero_right, (riceCounterRotation r t).norm_map]
  have hr := h.restrict_preimage (measurableSet_ball : MeasurableSet (Metric.ball (0 : ℂ) 1))
  rw [hpre] at hr
  exact hr.smul_measure (ENNReal.ofReal Real.pi⁻¹)

/-- The actual scale-dependent phase rotation of the original product probability parameters.
Source: arXiv:2412.09080v3, §2.1, the joint observation of the explicit field testing `thm:kac-rice`. -/
noncomputable def riceCounterRotationSkew (t : ℝ) (ω : ℝ × ℂ) : ℝ × ℂ :=
  (ω.1, riceCounterRotation ω.1 t ω.2)

/-- The actual scale-dependent disk rotation is measurable on the original product parameter space.
Source: arXiv:2412.09080v3, §2.1, the measurable joint-law construction for `thm:kac-rice`. -/
theorem measurable_riceCounterRotationSkew (t : ℝ) : Measurable (riceCounterRotationSkew t) := by
  have hz := measurable_snd.mul ((measurable_riceCounterPhase t).comp measurable_fst)
  change Measurable (fun ω : ℝ × ℂ => (ω.1, ω.2 * riceCounterPhase ω.1 t))
  exact measurable_fst.prodMk hz

/-- Fubini and actual disk rotation invariance show that the scale-dependent phase rotation preserves the entire original product probability law.
Source: arXiv:2412.09080v3, §2.1, the actual probability law of the joint observation in `thm:kac-rice`. -/
theorem measurePreserving_riceCounterRotationSkew (t : ℝ) :
    MeasurePreserving (riceCounterRotationSkew t) riceCounterLaw riceCounterLaw := by
  refine ⟨measurable_riceCounterRotationSkew t, ?_⟩
  apply Measure.ext
  intro B hB
  rw [Measure.map_apply (measurable_riceCounterRotationSkew t) hB, riceCounterLaw,
    Measure.prod_apply (hB.preimage (measurable_riceCounterRotationSkew t)), Measure.prod_apply hB]
  apply lintegral_congr_ae
  apply ae_of_all
  intro r
  have hslice : MeasurableSet (Prod.mk r ⁻¹' B) := hB.preimage (measurable_const.prodMk measurable_id)
  have hrot := measurePreserving_riceCounterRotation r t
  have heq := congrArg (fun μ : Measure ℂ => μ (Prod.mk r ⁻¹' B)) hrot.map_eq
  rw [Measure.map_apply hrot.measurable hslice] at heq
  exact heq

/-- The explicit time-zero joint map, retaining both the original field value and its scale-free derivative off the null zero scale.
Source: arXiv:2412.09080v3, §2.1, the concrete joint law used for hypothesis three of `thm:kac-rice`. -/
noncomputable def riceCounterJointBase (ω : ℝ × ℂ) : ℝ × ℝ :=
  (ω.1 * (2 + ω.2.re), -ω.2.im)

/-- The original time-zero joint map is measurable on the original product parameter space.
Source: arXiv:2412.09080v3, §2.1, the concrete joint-law map used for `thm:kac-rice`. -/
theorem measurable_riceCounterJointBase : Measurable riceCounterJointBase := by
  exact (measurable_fst.mul (measurable_const.add (Complex.reCLM.continuous.measurable.comp measurable_snd))).prodMk
    (Complex.imCLM.continuous.measurable.comp measurable_snd).neg

/-- The actual field and actual derivative observations form a measurable joint map at every fixed time.
Source: arXiv:2412.09080v3, §2.1, the measurable joint law in hypothesis three of `thm:kac-rice`. -/
theorem measurable_riceCounter_joint (t : ℝ) :
    Measurable (fun ω : ℝ × ℂ => (riceCounterField ω t, deriv (riceCounterField ω) t)) :=
  (measurable_riceCounterField t).prodMk (measurable_deriv_riceCounterField t)

/-- At every nonzero actual scale, the joint observation at time `t` is exactly the time-zero map after the actual disk phase rotation.
Source: arXiv:2412.09080v3, §2.1, the joint observation in hypothesis three of `thm:kac-rice`. -/
theorem riceCounter_joint_eq_base_rotation (ω : ℝ × ℂ) (hr : ω.1 ≠ 0) (t : ℝ) :
    (riceCounterField ω t, deriv (riceCounterField ω) t) = riceCounterJointBase (riceCounterRotationSkew t ω) := by
  simp only [deriv_riceCounterField, riceCounterDerivative_of_ne_zero ω hr, riceCounterField,
    riceCounterJointBase, riceCounterRotationSkew, riceCounterRotation_apply]

/-- The actual field and actual derivative have a time-independent joint law under the original product probability initialization.
Source: arXiv:2412.09080v3, §2.1, the actual joint distribution used to test hypothesis three of `thm:kac-rice`. -/
theorem map_riceCounter_joint_eq_base (t : ℝ) :
    riceCounterLaw.map (fun ω => (riceCounterField ω t, deriv (riceCounterField ω) t)) =
      riceCounterLaw.map riceCounterJointBase := by
  have heq : (fun ω => (riceCounterField ω t, deriv (riceCounterField ω) t)) =ᵐ[riceCounterLaw]
      riceCounterJointBase ∘ riceCounterRotationSkew t := by
    filter_upwards [ae_riceCounter_good] with ω hω
    exact riceCounter_joint_eq_base_rotation ω hω.1 t
  rw [Measure.map_congr heq, ← Measure.map_map measurable_riceCounterJointBase (measurable_riceCounterRotationSkew t),
    (measurePreserving_riceCounterRotationSkew t).map_eq]

/-- The actual field values alone have the first marginal of the same concrete time-zero joint law.
Source: arXiv:2412.09080v3, §2.1, the actual scalar law in hypothesis two of `thm:kac-rice`. -/
theorem map_riceCounterField_eq_base_fst (t : ℝ) :
    riceCounterLaw.map (fun ω => riceCounterField ω t) = (riceCounterLaw.map riceCounterJointBase).map Prod.fst := by
  rw [← map_riceCounter_joint_eq_base t, Measure.map_map measurable_fst
    ((measurable_riceCounterField t).prodMk (measurable_deriv_riceCounterField t))]
  rfl

/-- The actual derivative law is the second marginal of the same concrete time-zero joint law.
Source: arXiv:2412.09080v3, §2.1, the actual derivative distribution entering the Kac–Rice integral. -/
theorem map_riceCounterDerivative_eq_base_snd (t : ℝ) :
    riceCounterLaw.map (fun ω => deriv (riceCounterField ω) t) =
      (riceCounterLaw.map riceCounterJointBase).map Prod.snd := by
  rw [← map_riceCounter_joint_eq_base t,
    Measure.map_map measurable_snd (measurable_riceCounter_joint t)]
  rfl

/-- A genuine nonzero scale and moving disk coordinate jointly realize the phase-law equality at every real observation time.
Source: arXiv:2412.09080v3, §2.1, a concrete joint-law witness for the construction testing `thm:kac-rice`. -/
example (t : ℝ) : ((1 / 2 : ℝ) ≠ 0) ∧
    (riceCounterField ((1 / 2 : ℝ), Complex.I / 2) t,
      deriv (riceCounterField ((1 / 2 : ℝ), Complex.I / 2)) t) =
      riceCounterJointBase (riceCounterRotationSkew t ((1 / 2 : ℝ), Complex.I / 2)) :=
  ⟨by norm_num, riceCounter_joint_eq_base_rotation _ (by norm_num) t⟩

end Transformer.Modes
