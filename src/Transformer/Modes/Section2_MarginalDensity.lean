/-
# The actual scalar density of the oscillating field

Source: arXiv:2412.09080v3, §2.1, hypothesis two of
`thm:kac-rice`. Integrating the proved actual joint density
over the derivative coordinate gives the field-value density.

The joint density is bounded by `1/π` and vanishes outside
the derivative interval `(-1,1)`. Its marginal is therefore
finite everywhere, with bound `2/π`. Passing to a real
nonnegative density retains its exact measure.

Tonelli and the proved joint expectation identity establish
the scalar pushforward equality for every observation time.
The actual joint density is independent of the field value
throughout `(-1,1)`, so the marginal is constant there. This
gives the exact time-and-value continuity required by the
paper near level zero.

The scalar density is derived from the actual law, rather
than introduced only on the continuity neighborhood. The
positive constant test function checks the normalization
under the original probability measure.
-/

import Transformer.Modes.Section2_DensityContinuity

open Real Set Filter MeasureTheory
open scoped ENNReal

namespace Transformer.Modes

/-- The actual field-value marginal of the explicit joint density before its finite real conversion.
Source: arXiv:2412.09080v3, §2.1, the scalar law in hypothesis two of `thm:kac-rice`. -/
noncomputable def riceCounterValueDensityENN (x : ℝ) : ℝ≥0∞ :=
  ∫⁻ y, riceCounterJointDensityENN (x, y)

/-- The actual scalar marginal density is measurable.
Source: arXiv:2412.09080v3, §2.1, the actual scalar density required in hypothesis two. -/
theorem measurable_riceCounterValueDensityENN : Measurable riceCounterValueDensityENN :=
  measurable_riceCounterJointDensityENN.lintegral_prod_right'

/-- The actual joint density is dominated by its original derivative-support interval and its exact global bound.
Source: arXiv:2412.09080v3, §2.1, the finite-marginal domination for the actual scalar density. -/
theorem riceCounterJointDensityENN_le_indicator (x y : ℝ) :
    riceCounterJointDensityENN (x, y) ≤ (Ioo (-1 : ℝ) 1).indicator (fun _ => ENNReal.ofReal Real.pi⁻¹) y := by
  by_cases hy : y ∈ Ioo (-1 : ℝ) 1
  · rw [indicator_of_mem hy]
    exact riceCounterJointDensityENN_le (x, y)
  · rw [indicator_of_notMem hy, riceCounterJointDensityENN_zero_of_outside (x, y)
      (fun h => hy (abs_lt.mp h.2))]

/-- The actual scalar density has a finite global bound from the actual bounded derivative support.
Source: arXiv:2412.09080v3, §2.1, the finite real-valued scalar density in hypothesis two. -/
theorem riceCounterValueDensityENN_le (x : ℝ) :
    riceCounterValueDensityENN x ≤ ENNReal.ofReal (Real.pi⁻¹ * 2) := by
  calc
    _ ≤ ∫⁻ y, (Ioo (-1 : ℝ) 1).indicator (fun _ => ENNReal.ofReal Real.pi⁻¹) y :=
      lintegral_mono (fun y => riceCounterJointDensityENN_le_indicator x y)
    _ = _ := by
      rw [lintegral_indicator measurableSet_Ioo, setLIntegral_const, Real.volume_Ioo,
        ← ENNReal.ofReal_mul (inv_nonneg.mpr Real.pi_pos.le)]
      norm_num

/-- The actual scalar density is finite at every field value.
Source: arXiv:2412.09080v3, §2.1, the everywhere real-valued scalar density used for hypothesis two. -/
theorem riceCounterValueDensityENN_ne_top (x : ℝ) : riceCounterValueDensityENN x ≠ ⊤ :=
  ne_top_of_le_ne_top ENNReal.ofReal_ne_top (riceCounterValueDensityENN_le x)

/-- The actual real nonnegative scalar density retains the original field observation argument.
Source: arXiv:2412.09080v3, §2.1, the scalar density function in hypothesis two of `thm:kac-rice`. -/
noncomputable def riceCounterValueDensity (x : ℝ) : ℝ := (riceCounterValueDensityENN x).toReal

/-- The actual scalar real density is measurable.
Source: arXiv:2412.09080v3, §2.1, the scalar law's real density in hypothesis two. -/
theorem measurable_riceCounterValueDensity : Measurable riceCounterValueDensity :=
  measurable_riceCounterValueDensityENN.ennreal_toReal

/-- The actual scalar real density is nonnegative everywhere.
Source: arXiv:2412.09080v3, §2.1, the actual nonnegative scalar density in hypothesis two. -/
theorem riceCounterValueDensity_nonneg (x : ℝ) : 0 ≤ riceCounterValueDensity x := ENNReal.toReal_nonneg

/-- The actual finite marginal is exactly recovered from its real density.
Source: arXiv:2412.09080v3, §2.1, the exact scalar-density normalization in hypothesis two. -/
theorem ofReal_riceCounterValueDensity (x : ℝ) :
    ENNReal.ofReal (riceCounterValueDensity x) = riceCounterValueDensityENN x :=
  ENNReal.ofReal_toReal (riceCounterValueDensityENN_ne_top x)

/-- Every actual field-value expectation equals its actual scalar-density integral.
Source: arXiv:2412.09080v3, §2.1, the actual scalar-law identity required by hypothesis two. -/
theorem lintegral_riceCounterField_density (t : ℝ) (g : ℝ → ℝ≥0∞) (hg : Measurable g) :
    ∫⁻ ω, g (riceCounterField ω t) ∂riceCounterLaw = ∫⁻ x, riceCounterValueDensityENN x * g x := by
  have hf : Measurable (fun q : ℝ × ℝ => riceCounterJointDensityENN q * g q.1) :=
    measurable_riceCounterJointDensityENN.mul (hg.comp measurable_fst)
  calc
    _ = ∫⁻ q : ℝ × ℝ, g q.1 ∂riceCounterLaw.map riceCounterJointBase := by
      rw [← lintegral_map hg (measurable_riceCounterField t), map_riceCounterField_eq_base_fst t]
      exact lintegral_map hg measurable_fst
    _ = ∫⁻ q : ℝ × ℝ, riceCounterJointDensityENN q * g q.1 := by
      exact (lintegral_map (f := fun q : ℝ × ℝ => g q.1) (hg.comp measurable_fst)
        measurable_riceCounterJointBase).trans
          (lintegral_riceCounterJointBase_density _ (hg.comp measurable_fst))
    _ = ∫⁻ x, ∫⁻ y, riceCounterJointDensityENN (x, y) * g x :=
      lintegral_prod _ hf.aemeasurable
    _ = _ := by
      apply lintegral_congr_ae
      apply ae_of_all
      intro x
      exact lintegral_mul_const _ (show Measurable (fun y => riceCounterJointDensityENN (x, y)) from
        measurable_riceCounterJointDensityENN.comp (measurable_const.prodMk measurable_id))

/-- The actual field-value pushforward has the displayed actual real density at every real observation time.
Source: arXiv:2412.09080v3, §2.1, the exact map equality required in hypothesis two of `thm:kac-rice`. -/
theorem map_riceCounterField_withDensity (t : ℝ) :
    riceCounterLaw.map (fun ω => riceCounterField ω t) =
      (volume : Measure ℝ).withDensity (fun x => ENNReal.ofReal (riceCounterValueDensity x)) := by
  simp_rw [ofReal_riceCounterValueDensity]
  apply Measure.ext
  intro B hB
  have hg : Measurable (B.indicator (fun _ => (1 : ℝ≥0∞))) := measurable_const.indicator hB
  have h := lintegral_riceCounterField_density t _ hg
  have he := (lintegral_map hg (measurable_riceCounterField t)).trans
    (h.trans (lintegral_withDensity_eq_lintegral_mul volume measurable_riceCounterValueDensityENN hg).symm)
  simpa only [lintegral_indicator_fun_one hB] using he

/-- The actual scalar density is constant throughout the original nontrivial field-value neighborhood of zero.
Source: arXiv:2412.09080v3, §2.1, the actual local scalar-density property giving hypothesis two's continuity. -/
theorem riceCounterValueDensity_near (x : ℝ) (hx : |x| < 1) :
    riceCounterValueDensity x = riceCounterValueDensity 0 := by
  unfold riceCounterValueDensity riceCounterValueDensityENN
  congr 1
  apply lintegral_congr_ae
  apply ae_of_all
  intro y
  change riceCounterJointDensityENN (x, y) = riceCounterJointDensityENN (0, y)
  rw [riceCounterJointDensityENN_near x y hx, riceCounterJointDensityENN_near 0 y (by norm_num)]

/-- The actual scalar density is continuous jointly in time and field value on the required neighborhood.
Source: arXiv:2412.09080v3, §2.1, the exact continuity condition of hypothesis two of `thm:kac-rice`. -/
theorem continuousOn_riceCounterValueDensity (T : Set ℝ) :
    ContinuousOn (fun q : ℝ × ℝ => riceCounterValueDensity q.2) (T ×ˢ Ioo (-1 : ℝ) 1) := by
  apply continuousOn_const.congr
  intro q hq
  exact riceCounterValueDensity_near q.2 (abs_lt.mpr hq.2)

/-- The positive constant test function gives the exact probability normalization of the actual scalar density.
Source: arXiv:2412.09080v3, §2.1, a nonzero example satisfying the scalar expectation identity's hypotheses. -/
example : Measurable (fun _ : ℝ => (1 : ℝ≥0∞)) ∧ (∫⁻ x, riceCounterValueDensityENN x) = 1 := by
  refine ⟨measurable_const, ?_⟩
  have h := lintegral_riceCounterField_density 0 (fun _ => 1) measurable_const
  simpa only [mul_one, lintegral_one, measure_univ] using h.symm

/-- A nonzero actual field value jointly realizes the local-density hypothesis and the proved exact equality.
Source: arXiv:2412.09080v3, §2.1, a genuine near-zero scalar-density example. -/
example : |(1 / 2 : ℝ)| < 1 ∧ riceCounterValueDensity (1 / 2) = riceCounterValueDensity 0 :=
  ⟨by norm_num, riceCounterValueDensity_near _ (by norm_num)⟩

end Transformer.Modes
