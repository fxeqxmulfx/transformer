/-
# The actual finite joint density of the oscillating field

Source: arXiv:2412.09080v3, §2.1, the joint-law condition
in `thm:kac-rice`. Integrating the explicit slice kernel
against its latent real disk coordinate gives the actual
joint density of the field and its derivative.

Tonelli changes the order of the three nonnegative integrals
in the proved expectation identity. The resulting equality
holds for every measurable nonnegative test function and
therefore identifies the pushforward measure itself. The
time-invariance result transfers this density from the base
joint map to the actual field and actual derivative at every
real observation time.

The kernel's interval domination bounds its latent integral
by one and the joint density by `1/π`. Thus the density is
finite everywhere, and converting it to a nonnegative real
function loses no mass. The real version is the form needed
in the paper's hypothesis. Continuity near level zero is a
separate property, not presumed in this calculation.
-/

import Transformer.Modes.Section2_SliceKernel

open Real Set Filter MeasureTheory
open scoped ENNReal

namespace Transformer.Modes

/-- The actual joint density before converting its proved finite values to real numbers.
Source: arXiv:2412.09080v3, §2.1, the disk-slice formula deriving hypothesis three of `thm:kac-rice`. -/
noncomputable def riceCounterJointDensityENN (q : ℝ × ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal Real.pi⁻¹ * ∫⁻ u, riceCounterSliceKernel u q.1 q.2

/-- The actual joint density is a measurable function of both observation coordinates.
Source: arXiv:2412.09080v3, §2.1, the actual density required by hypothesis three of `thm:kac-rice`. -/
theorem measurable_riceCounterJointDensityENN : Measurable riceCounterJointDensityENN := by
  exact measurable_const.mul measurable_riceCounterSliceKernel.lintegral_prod_left'

/-- The actual latent-coordinate density integral is bounded by one at every observation point.
Source: arXiv:2412.09080v3, §2.1, the boundedness establishing finiteness of the actual joint density. -/
theorem lintegral_riceCounterSliceKernel_le_one (q : ℝ × ℝ) :
    (∫⁻ u, riceCounterSliceKernel u q.1 q.2) ≤ 1 := by
  calc
    _ ≤ ∫⁻ u, (Ioo (-1 : ℝ) 1).indicator (fun _ => ENNReal.ofReal (1 / 2 : ℝ)) u :=
      lintegral_mono (fun u => riceCounterSliceKernel_le u q.1 q.2)
    _ = 1 := by
      rw [lintegral_indicator measurableSet_Ioo, setLIntegral_const, Real.volume_Ioo]
      rw [← ENNReal.ofReal_mul (by norm_num : (0 : ℝ) ≤ 1 / 2)]
      norm_num

/-- The actual joint density has a global finite bound with the original disk normalization.
Source: arXiv:2412.09080v3, §2.1, the pointwise finite actual density in hypothesis three of `thm:kac-rice`. -/
theorem riceCounterJointDensityENN_le (q : ℝ × ℝ) :
    riceCounterJointDensityENN q ≤ ENNReal.ofReal Real.pi⁻¹ := by
  calc
    _ ≤ ENNReal.ofReal Real.pi⁻¹ * 1 := by
      unfold riceCounterJointDensityENN
      gcongr
      exact lintegral_riceCounterSliceKernel_le_one q
    _ = _ := mul_one _

/-- The actual joint density is finite everywhere, not merely almost everywhere.
Source: arXiv:2412.09080v3, §2.1, the finite real-valued density required for the continuity hypothesis. -/
theorem riceCounterJointDensityENN_ne_top (q : ℝ × ℝ) : riceCounterJointDensityENN q ≠ ⊤ :=
  ne_top_of_le_ne_top ENNReal.ofReal_ne_top (riceCounterJointDensityENN_le q)

/-- The real actual joint density retains both observation arguments through the explicit slice integral.
Source: arXiv:2412.09080v3, §2.1, the density function in hypothesis three of `thm:kac-rice`. -/
noncomputable def riceCounterJointDensity (q : ℝ × ℝ) : ℝ :=
  (riceCounterJointDensityENN q).toReal

/-- The actual real joint density is measurable.
Source: arXiv:2412.09080v3, §2.1, the actual density of the joint field observation. -/
theorem measurable_riceCounterJointDensity : Measurable riceCounterJointDensity :=
  measurable_riceCounterJointDensityENN.ennreal_toReal

/-- The actual real joint density is nonnegative at every observation point.
Source: arXiv:2412.09080v3, §2.1, the nonnegative density entering the Kac–Rice integral. -/
theorem riceCounterJointDensity_nonneg (q : ℝ × ℝ) : 0 ≤ riceCounterJointDensity q :=
  ENNReal.toReal_nonneg

/-- Converting the actual real density back to a nonnegative extended real recovers the exact slice density everywhere.
Source: arXiv:2412.09080v3, §2.1, the exact real-valued density identifying the joint law. -/
theorem ofReal_riceCounterJointDensity (q : ℝ × ℝ) :
    ENNReal.ofReal (riceCounterJointDensity q) = riceCounterJointDensityENN q :=
  ENNReal.ofReal_toReal (riceCounterJointDensityENN_ne_top q)

/-- The actual base joint expectation equals integration against the explicit finite density for every nonnegative measurable test function.
Source: arXiv:2412.09080v3, §2.1, the full actual joint-law identity for hypothesis three of `thm:kac-rice`. -/
theorem lintegral_riceCounterJointBase_density (G : ℝ × ℝ → ℝ≥0∞) (hG : Measurable G) :
    ∫⁻ ω, G (riceCounterJointBase ω) ∂riceCounterLaw =
      ∫⁻ q, riceCounterJointDensityENN q * G q := by
  have hf : Measurable (fun p : ℝ × (ℝ × ℝ) => riceCounterSliceKernel p.1 p.2.1 p.2.2 * G p.2) :=
    measurable_riceCounterSliceKernel.mul (hG.comp measurable_snd)
  have hfu (u : ℝ) : Measurable (fun q : ℝ × ℝ => riceCounterSliceKernel u q.1 q.2 * G q) :=
    hf.comp (measurable_const.prodMk measurable_id)
  calc
    _ = ENNReal.ofReal Real.pi⁻¹ * ∫⁻ u, ∫⁻ y, ∫⁻ x,
        riceCounterSliceKernel u x y * G (x, y) := lintegral_riceCounterJointBase_kernel G hG
    _ = ENNReal.ofReal Real.pi⁻¹ * ∫⁻ u, ∫⁻ q : ℝ × ℝ,
        riceCounterSliceKernel u q.1 q.2 * G q := by
      congr 1
      apply lintegral_congr_ae
      apply ae_of_all
      intro u
      exact (lintegral_prod_symm (μ := (volume : Measure ℝ)) (ν := (volume : Measure ℝ))
        _ (hfu u).aemeasurable).symm
    _ = ENNReal.ofReal Real.pi⁻¹ * ∫⁻ q : ℝ × ℝ, ∫⁻ u,
        riceCounterSliceKernel u q.1 q.2 * G q := by
      congr 1
      exact lintegral_lintegral_swap (f := fun u q => riceCounterSliceKernel u q.1 q.2 * G q) hf.aemeasurable
    _ = ∫⁻ q : ℝ × ℝ, ENNReal.ofReal Real.pi⁻¹ * (∫⁻ u, riceCounterSliceKernel u q.1 q.2 * G q) :=
      (lintegral_const_mul _ hf.lintegral_prod_left').symm
    _ = _ := by
      apply lintegral_congr_ae
      apply ae_of_all
      intro q
      have hK : Measurable (fun u => riceCounterSliceKernel u q.1 q.2) :=
        measurable_riceCounterSliceKernel.comp (measurable_id.prodMk measurable_const)
      change ENNReal.ofReal Real.pi⁻¹ * (∫⁻ u, riceCounterSliceKernel u q.1 q.2 * G q) =
        riceCounterJointDensityENN q * G q
      rw [lintegral_mul_const _ hK, riceCounterJointDensityENN, mul_assoc]

/-- The pushforward of the actual base joint map has the explicit slice density with respect to real product volume.
Source: arXiv:2412.09080v3, §2.1, the actual joint-law density required by `thm:kac-rice`. -/
theorem map_riceCounterJointBase_withDensity :
    riceCounterLaw.map riceCounterJointBase = (volume : Measure (ℝ × ℝ)).withDensity riceCounterJointDensityENN := by
  apply Measure.ext
  intro B hB
  have hG : Measurable (B.indicator (fun _ => (1 : ℝ≥0∞))) := measurable_const.indicator hB
  have h := lintegral_riceCounterJointBase_density _ hG
  have he : (∫⁻ q, B.indicator (fun _ => (1 : ℝ≥0∞)) q ∂riceCounterLaw.map riceCounterJointBase) =
      ∫⁻ q, B.indicator (fun _ => (1 : ℝ≥0∞)) q ∂(volume : Measure (ℝ × ℝ)).withDensity riceCounterJointDensityENN :=
    (lintegral_map hG measurable_riceCounterJointBase).trans
      (h.trans (lintegral_withDensity_eq_lintegral_mul volume measurable_riceCounterJointDensityENN hG).symm)
  simpa only [lintegral_indicator_fun_one hB] using he

/-- The actual field and its actual derivative have the displayed real density at every observation time under the actual probability law.
Source: arXiv:2412.09080v3, §2.1, the actual map equality in hypothesis three of `thm:kac-rice`. -/
theorem map_riceCounter_joint_withDensity (t : ℝ) :
    riceCounterLaw.map (fun ω => (riceCounterField ω t, deriv (riceCounterField ω) t)) =
      (volume : Measure (ℝ × ℝ)).withDensity (fun q => ENNReal.ofReal (riceCounterJointDensity q)) := by
  simp_rw [ofReal_riceCounterJointDensity]
  rw [map_riceCounter_joint_eq_base t, map_riceCounterJointBase_withDensity]

/-- A positive measurable test function jointly realizes the exact density identity and the genuine probability normalization.
Source: arXiv:2412.09080v3, §2.1, a nonzero example of the full actual joint-law identity. -/
example : Measurable (fun _ : ℝ × ℝ => (1 : ℝ≥0∞)) ∧
    (∫⁻ q, riceCounterJointDensityENN q) = 1 := by
  refine ⟨measurable_const, ?_⟩
  have h := lintegral_riceCounterJointBase_density (fun _ => 1) measurable_const
  simpa only [mul_one, lintegral_one, measure_univ] using h.symm

end Transformer.Modes
