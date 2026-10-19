module

public import Transformer.Modes.Section2_SliceWidth
public import Mathlib.MeasureTheory.Integral.DominatedConvergence

/-!
# Continuity and positivity of the actual local joint density

Source: arXiv:2412.09080v3, §2.1, hypothesis three and
the right side of `eq:krf`. For field values `|x|<1`, the
actual finite joint density is the normalized integral of
the positive scale reciprocal across the actual disk slice.

The slice endpoints are continuous functions of the derivative
coordinate, including where the slice disappears. The
integrand has an everywhere continuous positive extension
that agrees with the actual reciprocal on the disk. Writing
the moving interval integral as the difference of two values
of its continuous primitive proves continuity for every real
derivative coordinate.

The actual density equals this continuous slice function
throughout a genuine open neighborhood of field level zero.
It is strictly positive whenever the derivative coordinate
has magnitude below one. Both facts concern the density
already proved to identify the actual joint law, rather than
an independently chosen candidate density.
-/

@[expose] public section

open Real Set Filter MeasureTheory
open scoped ENNReal

namespace Transformer.Modes

/-- The actual local joint density as a normalized real interval integral across the disk slice.
Source: arXiv:2412.09080v3, §2.1, the actual joint density near level zero in hypothesis three of `thm:kac-rice`. -/
noncomputable def riceCounterSliceDensity (y : ℝ) : ℝ :=
  Real.pi⁻¹ * ∫ u in (-riceCounterSliceWidth y)..riceCounterSliceWidth y, riceCounterSliceReciprocal u

/-- The actual real slice integral is nonnegative at every derivative coordinate.
Source: arXiv:2412.09080v3, §2.1, nonnegativity of the actual density in `eq:krf`. -/
theorem riceCounterSliceIntegral_nonneg (y : ℝ) :
    0 ≤ ∫ u in (-riceCounterSliceWidth y)..riceCounterSliceWidth y, riceCounterSliceReciprocal u := by
  rw [intervalIntegral.integral_of_le (by linarith [riceCounterSliceWidth_nonneg y])]
  exact integral_nonneg (fun u => (riceCounterSliceReciprocal_pos u).le)

/-- The exact nonnegative slice integral equals the nonnegative extension of the actual real interval integral.
Source: arXiv:2412.09080v3, §2.1, the exact real conversion of the actual joint-density formula. -/
theorem lintegral_riceCounterSliceReciprocal (y : ℝ) :
    (∫⁻ u in Ioo (-riceCounterSliceWidth y) (riceCounterSliceWidth y),
      ENNReal.ofReal (riceCounterSliceReciprocal u)) =
      ENNReal.ofReal (∫ u in (-riceCounterSliceWidth y)..riceCounterSliceWidth y, riceCounterSliceReciprocal u) := by
  rw [intervalIntegral.integral_of_le (by linarith [riceCounterSliceWidth_nonneg y]), integral_Ioc_eq_integral_Ioo]
  exact (ofReal_integral_eq_lintegral_ofReal
    (continuous_riceCounterSliceReciprocal.integrableOn_Icc.mono_set Ioo_subset_Icc_self)
    (ae_of_all _ (fun u => (riceCounterSliceReciprocal_pos u).le))).symm

/-- The actual real joint density equals the slice function throughout the original nontrivial field-value neighborhood.
Source: arXiv:2412.09080v3, §2.1, the exact actual density verifying hypothesis three of `thm:kac-rice`. -/
theorem riceCounterJointDensity_near (x y : ℝ) (hx : |x| < 1) :
    riceCounterJointDensity (x, y) = riceCounterSliceDensity y := by
  rw [riceCounterJointDensity, riceCounterJointDensityENN_near x y hx,
    lintegral_riceCounterSliceReciprocal, ENNReal.toReal_mul,
    ENNReal.toReal_ofReal (inv_nonneg.mpr Real.pi_pos.le),
    ENNReal.toReal_ofReal (riceCounterSliceIntegral_nonneg y)]
  rfl

/-- The actual slice density is the difference of two continuous primitive values at its exact moving endpoints.
Source: arXiv:2412.09080v3, §2.1, the continuity argument for the actual joint density. -/
theorem riceCounterSliceDensity_eq_primitive (y : ℝ) :
    riceCounterSliceDensity y = Real.pi⁻¹ *
      ((∫ u in (0 : ℝ)..riceCounterSliceWidth y, riceCounterSliceReciprocal u) -
        ∫ u in (0 : ℝ)..(-riceCounterSliceWidth y), riceCounterSliceReciprocal u) := by
  have h := intervalIntegral.integral_add_adjacent_intervals (μ := (volume : Measure ℝ))
    (continuous_riceCounterSliceReciprocal.intervalIntegrable 0 (-riceCounterSliceWidth y))
    (continuous_riceCounterSliceReciprocal.intervalIntegrable (-riceCounterSliceWidth y) (riceCounterSliceWidth y))
  unfold riceCounterSliceDensity
  congr 1
  linarith

/-- The actual slice density is continuous for every real derivative coordinate, including support-boundary points.
Source: arXiv:2412.09080v3, §2.1, the full derivative-coordinate continuity required by hypothesis three of `thm:kac-rice`. -/
theorem continuous_riceCounterSliceDensity : Continuous riceCounterSliceDensity := by
  have hprim : Continuous (fun b : ℝ => ∫ u in (0 : ℝ)..b, riceCounterSliceReciprocal u) :=
    intervalIntegral.continuous_primitive (fun a b => continuous_riceCounterSliceReciprocal.intervalIntegrable a b) 0
  have he : riceCounterSliceDensity = fun y => Real.pi⁻¹ *
      ((∫ u in (0 : ℝ)..riceCounterSliceWidth y, riceCounterSliceReciprocal u) -
        ∫ u in (0 : ℝ)..(-riceCounterSliceWidth y), riceCounterSliceReciprocal u) :=
    funext riceCounterSliceDensity_eq_primitive
  rw [he]
  exact continuous_const.mul ((hprim.comp continuous_riceCounterSliceWidth).sub
    (hprim.comp continuous_riceCounterSliceWidth.neg))

/-- The actual joint density is continuous in time, the field-value neighborhood, and every derivative coordinate.
Source: arXiv:2412.09080v3, §2.1, the exact continuity domain in hypothesis three of `thm:kac-rice`. -/
theorem continuousOn_riceCounterJointDensity (T : Set ℝ) :
    ContinuousOn (fun q : ℝ × ℝ × ℝ => riceCounterJointDensity (q.2.1, q.2.2))
      (T ×ˢ Ioo (-1 : ℝ) 1 ×ˢ (univ : Set ℝ)) := by
  apply (continuous_riceCounterSliceDensity.comp continuous_snd.snd).continuousOn.congr
  intro q hq
  exact riceCounterJointDensity_near q.2.1 q.2.2 (abs_lt.mpr hq.2.1)

/-- The actual slice density is strictly positive throughout the interior derivative support.
Source: arXiv:2412.09080v3, §2.1, the positive actual density needed for the right side of `eq:krf`. -/
theorem riceCounterSliceDensity_pos (y : ℝ) (hy : |y| < 1) : 0 < riceCounterSliceDensity y := by
  have hw := (riceCounterSliceWidth_pos_iff y).mpr hy
  unfold riceCounterSliceDensity
  apply mul_pos (inv_pos.mpr Real.pi_pos)
  apply intervalIntegral.integral_pos (by linarith) continuous_riceCounterSliceReciprocal.continuousOn
  · exact fun u hu => (riceCounterSliceReciprocal_pos u).le
  · exact ⟨0, ⟨by linarith, by linarith⟩, riceCounterSliceReciprocal_pos 0⟩

/-- The actual joint density is strictly positive near level zero at every interior derivative coordinate.
Source: arXiv:2412.09080v3, §2.1, positivity of the actual density in the printed Kac–Rice integral. -/
theorem riceCounterJointDensity_pos (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
    0 < riceCounterJointDensity (x, y) := by
  rw [riceCounterJointDensity_near x y hx]
  exact riceCounterSliceDensity_pos y hy

/-- The actual slice density is nonnegative on the entire derivative-coordinate line.
Source: arXiv:2412.09080v3, §2.1, the nonnegative actual joint density in `eq:krf`. -/
theorem riceCounterSliceDensity_nonneg (y : ℝ) : 0 ≤ riceCounterSliceDensity y :=
  mul_nonneg (inv_nonneg.mpr Real.pi_pos.le) (riceCounterSliceIntegral_nonneg y)

/-- The actual continuous slice density retains the global bound from the proved joint law.
Source: arXiv:2412.09080v3, §2.1, the finite actual density in hypothesis three of `thm:kac-rice`. -/
theorem riceCounterSliceDensity_le (y : ℝ) : riceCounterSliceDensity y ≤ Real.pi⁻¹ := by
  rw [← riceCounterJointDensity_near 0 y (by norm_num), riceCounterJointDensity]
  exact (ENNReal.toReal_mono ENNReal.ofReal_ne_top (riceCounterJointDensityENN_le (0, y))).trans_eq
    (ENNReal.toReal_ofReal (inv_nonneg.mpr Real.pi_pos.le))

/-- At the original zero field level, the actual joint density is exactly the continuous slice density.
Source: arXiv:2412.09080v3, §2.1, the actual density on the level used in the Kac–Rice integral. -/
theorem riceCounterJointDensity_zero_level (y : ℝ) : riceCounterJointDensity (0, y) = riceCounterSliceDensity y :=
  riceCounterJointDensity_near 0 y (by norm_num)

/-- The actual continuous slice density is zero at and beyond both derivative-support endpoints.
Source: arXiv:2412.09080v3, §2.1, the actual density on the full derivative-coordinate continuity domain. -/
theorem riceCounterSliceDensity_eq_zero (y : ℝ) (hy : 1 ≤ |y|) : riceCounterSliceDensity y = 0 := by
  rw [← riceCounterJointDensity_zero_level]
  exact riceCounterJointDensity_zero_of_outside (0, y) (fun h => (not_lt_of_ge hy) h.2)

/-- A genuine nonzero value and derivative jointly realize all local-density and positivity hypotheses.
Source: arXiv:2412.09080v3, §2.1, a positive actual-density example in the original continuity neighborhood. -/
example : |(1 / 2 : ℝ)| < 1 ∧ |(1 / 2 : ℝ)| < 1 ∧
    riceCounterJointDensity (1 / 2, 1 / 2) = riceCounterSliceDensity (1 / 2) ∧
    0 < riceCounterSliceDensity (1 / 2) ∧ 0 < riceCounterJointDensity (1 / 2, 1 / 2) := by
  exact ⟨by norm_num, by norm_num, riceCounterJointDensity_near _ _ (by norm_num),
    riceCounterSliceDensity_pos _ (by norm_num), riceCounterJointDensity_pos _ _ (by norm_num) (by norm_num)⟩

/-- A genuine derivative-support boundary point realizes the vanishing hypothesis and conclusion.
Source: arXiv:2412.09080v3, §2.1, an actual support-boundary example for the continuous density. -/
example : 1 ≤ |(1 : ℝ)| ∧ riceCounterSliceDensity 1 = 0 :=
  ⟨by norm_num, riceCounterSliceDensity_eq_zero 1 (by norm_num)⟩

end Transformer.Modes
