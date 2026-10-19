module

public import Transformer.Modes.Section2_DiskCoordinates

/-!
# The actual joint-law slice kernel

Source: arXiv:2412.09080v3, §2.1, hypothesis three of
`thm:kac-rice`. The disk-coordinate and positive-scale
substitutions give an explicit nonnegative kernel for the
time-zero joint observation of the oscillating field.

The real coordinate `u` is retained as a latent disk slice.
The kernel is zero outside `u²+y²<1` or outside the actual
stretched scale interval `|x|<2+u`. Inside this support it
equals the signed-scale density `1/(2*(2+u))`. The disk
normalization `1/π` is retained outside the integral.

The exact expectation identity follows from Tonelli and
the already proved changes of variables. It holds for every
nonnegative measurable test function. The kernel is bounded
by one half and vanishes outside a bounded `u` interval;
these facts will give a finite actual joint density when the
latent coordinate is integrated out. No final joint density
is assumed here.
-/

@[expose] public section

open Real Set Filter MeasureTheory
open scoped ENNReal

namespace Transformer.Modes

/-- The actual scale density on a disk slice, with the original disk and stretched-interval support retained.
Source: arXiv:2412.09080v3, §2.1, the slice kernel deriving the joint law in hypothesis three of `thm:kac-rice`. -/
noncomputable def riceCounterSliceKernel (u x y : ℝ) : ℝ≥0∞ := by
  classical
  exact if (u, y) ∈ riceCounterDiskDomain ∧ x ∈ Ioo (-(2 + u)) (2 + u)
    then ENNReal.ofReal (1 / (2 * (2 + u))) else 0

/-- The actual slice kernel is jointly measurable in the latent disk coordinate and both observation coordinates.
Source: arXiv:2412.09080v3, §2.1, the measurable integral defining the actual joint density. -/
theorem measurable_riceCounterSliceKernel :
    Measurable (fun p : ℝ × (ℝ × ℝ) => riceCounterSliceKernel p.1 p.2.1 p.2.2) := by
  unfold riceCounterSliceKernel
  apply Measurable.ite ?_ (by fun_prop) measurable_const
  exact (measurableSet_riceCounterDiskDomain.preimage
    (measurable_fst.prodMk measurable_snd.snd)).inter
      ((measurableSet_lt (by fun_prop) measurable_snd.fst).inter
        (measurableSet_lt measurable_snd.fst (by fun_prop)))

/-- The actual slice kernel is bounded by one half and vanishes outside the original latent interval.
Source: arXiv:2412.09080v3, §2.1, the bounded integrable domination for the actual joint density. -/
theorem riceCounterSliceKernel_le (u x y : ℝ) :
    riceCounterSliceKernel u x y ≤ (Ioo (-1 : ℝ) 1).indicator (fun _ => ENNReal.ofReal (1 / 2 : ℝ)) u := by
  by_cases h : (u, y) ∈ riceCounterDiskDomain ∧ x ∈ Ioo (-(2 + u)) (2 + u)
  · have hb := riceCounterDiskDomain_stretch_bounds (u, y) h.1
    have hu : u ∈ Ioo (-1 : ℝ) 1 := abs_lt.mp (riceCounterDiskDomain_bounds (u, y) h.1).1
    rw [riceCounterSliceKernel, ite_eq_left h, indicator_of_mem hu]
    apply ENNReal.ofReal_le_ofReal
    apply (div_le_iff₀ (by linarith : 0 < 2 * (2 + u))).mpr
    nlinarith [hb.1]
  · rw [riceCounterSliceKernel, ite_eq_right h]
    exact zero_le

/-- Every nonzero slice-kernel value lies in the actual bounded observation support.
Source: arXiv:2412.09080v3, §2.1, the bounded support of the actual joint density. -/
theorem riceCounterSliceKernel_ne_zero_bounds (u x y : ℝ) (hK : riceCounterSliceKernel u x y ≠ 0) :
    |x| < 3 ∧ |y| < 1 := by
  by_cases h : (u, y) ∈ riceCounterDiskDomain ∧ x ∈ Ioo (-(2 + u)) (2 + u)
  · have hx : |x| < 2 + u := abs_lt.mpr h.2
    exact ⟨hx.trans (riceCounterDiskDomain_stretch_bounds (u, y) h.1).2,
      (riceCounterDiskDomain_bounds (u, y) h.1).2.1⟩
  · simp only [riceCounterSliceKernel, ite_eq_right h, ne_eq, not_true_eq_false] at hK

/-- The actual joint-map expectation is the normalized disk integral of the exact stretched-scale expectation.
Source: arXiv:2412.09080v3, §2.1, the actual changes of variables deriving hypothesis three of `thm:kac-rice`. -/
theorem lintegral_riceCounterJointBase_disk (G : ℝ × ℝ → ℝ≥0∞) (hG : Measurable G) :
    ∫⁻ ω, G (riceCounterJointBase ω) ∂riceCounterLaw =
      ENNReal.ofReal Real.pi⁻¹ * ∫⁻ q in riceCounterDiskDomain,
        ENNReal.ofReal (1 / (2 * (2 + q.1))) * ∫⁻ x in Ioo (-(2 + q.1)) (2 + q.1), G (x, q.2) := by
  have hparam : Measurable (fun p : (ℝ × ℝ) × ℝ => G (p.2 * (2 + p.1.1), p.1.2)) :=
    hG.comp ((measurable_snd.mul (measurable_const.add measurable_fst.fst)).prodMk measurable_fst.snd)
  have hinner := hparam.lintegral_prod_right' (ν := riceCounterScaleLaw)
  calc
    _ = ∫⁻ z, ∫⁻ r, G (riceCounterJointBase (r, z)) ∂riceCounterScaleLaw ∂riceCounterDiskLaw :=
      lintegral_prod_symm _ (hG.comp measurable_riceCounterJointBase).aemeasurable
    _ = ENNReal.ofReal Real.pi⁻¹ * ∫⁻ q in riceCounterDiskDomain,
        ∫⁻ r, G (r * (2 + q.1), q.2) ∂riceCounterScaleLaw :=
      lintegral_riceCounterDisk_coords _ hinner
    _ = _ := by
      congr 1
      apply setLIntegral_congr_fun measurableSet_riceCounterDiskDomain
      intro q hq
      have hc := (riceCounterDiskDomain_bounds q hq).2.2
      simpa only [mul_comm] using lintegral_riceCounterScale_stretch (2 + q.1) hc
        (fun x => G (x, q.2)) (hG.comp (measurable_id.prodMk measurable_const))

/-- Integrating the actual slice kernel against a test function gives precisely the disk-restricted stretched-scale integral.
Source: arXiv:2412.09080v3, §2.1, the exact support masks in the actual joint-density calculation. -/
theorem lintegral_riceCounterSliceKernel (q : ℝ × ℝ) (G : ℝ × ℝ → ℝ≥0∞) (hG : Measurable G) :
    (∫⁻ x, riceCounterSliceKernel q.1 x q.2 * G (x, q.2)) =
      riceCounterDiskDomain.indicator (fun q : ℝ × ℝ => ENNReal.ofReal (1 / (2 * (2 + q.1))) *
        ∫⁻ x in Ioo (-(2 + q.1)) (2 + q.1), G (x, q.2)) q := by
  by_cases hq : q ∈ riceCounterDiskDomain
  · rw [indicator_of_mem hq]
    have he : (fun x => riceCounterSliceKernel q.1 x q.2 * G (x, q.2)) =
        (Ioo (-(2 + q.1)) (2 + q.1)).indicator
          (fun x => ENNReal.ofReal (1 / (2 * (2 + q.1))) * G (x, q.2)) := by
      funext x
      by_cases hx : x ∈ Ioo (-(2 + q.1)) (2 + q.1)
      · simp only [riceCounterSliceKernel, hq, hx, and_self, ite_eq_left, indicator_of_mem hx]
      · simp only [riceCounterSliceKernel, hq, hx, and_false, ite_false, zero_mul, indicator_of_notMem hx]
    rw [he, lintegral_indicator measurableSet_Ioo]
    exact lintegral_const_mul _ (show Measurable (fun x : ℝ => G (x, q.2)) from
      hG.comp (measurable_id.prodMk measurable_const))
  · simp [riceCounterSliceKernel, hq]

/-- Tonelli gives the exact triple-integral representation of the actual joint law, retaining the explicit bounded slice kernel.
Source: arXiv:2412.09080v3, §2.1, the actual joint-density derivation for `thm:kac-rice`. -/
theorem lintegral_riceCounterJointBase_kernel (G : ℝ × ℝ → ℝ≥0∞) (hG : Measurable G) :
    ∫⁻ ω, G (riceCounterJointBase ω) ∂riceCounterLaw =
      ENNReal.ofReal Real.pi⁻¹ * ∫⁻ u, ∫⁻ y, ∫⁻ x,
        riceCounterSliceKernel u x y * G (x, y) := by
  have hparam : Measurable (fun p : (ℝ × ℝ) × ℝ => riceCounterSliceKernel p.1.1 p.2 p.1.2 * G (p.2, p.1.2)) :=
    (measurable_riceCounterSliceKernel.comp
      (measurable_fst.fst.prodMk (measurable_snd.prodMk measurable_fst.snd))).mul
        (hG.comp (measurable_snd.prodMk measurable_fst.snd))
  have hinner := hparam.lintegral_prod_right' (ν := (volume : Measure ℝ))
  have he : (∫⁻ q : ℝ × ℝ, ∫⁻ x, riceCounterSliceKernel q.1 x q.2 * G (x, q.2)) =
      ∫⁻ q in riceCounterDiskDomain, ENNReal.ofReal (1 / (2 * (2 + q.1))) *
        ∫⁻ x in Ioo (-(2 + q.1)) (2 + q.1), G (x, q.2) := by
    simp_rw [lintegral_riceCounterSliceKernel _ G hG]
    exact lintegral_indicator measurableSet_riceCounterDiskDomain _
  rw [lintegral_riceCounterJointBase_disk G hG, ← he]
  congr 1
  exact lintegral_prod (μ := (volume : Measure ℝ)) (ν := (volume : Measure ℝ))
    (fun q : ℝ × ℝ => ∫⁻ x, riceCounterSliceKernel q.1 x q.2 * G (x, q.2)) hinner.aemeasurable

/-- A genuine nonzero latent coordinate and interior observation realize the nonzero branch of the actual kernel.
Source: arXiv:2412.09080v3, §2.1, a positive-support example for the actual joint-law kernel. -/
example : riceCounterSliceKernel (1 / 2) (1 / 2) (1 / 2) = ENNReal.ofReal (1 / 5 : ℝ) := by
  norm_num [riceCounterSliceKernel, riceCounterDiskDomain]

/-- The same interior observation realizes the nonzero-kernel hypothesis and both actual support conclusions jointly.
Source: arXiv:2412.09080v3, §2.1, a genuine support witness for the actual density kernel. -/
example : riceCounterSliceKernel (1 / 2) (1 / 2) (1 / 2) ≠ 0 ∧
    |(1 / 2 : ℝ)| < 3 ∧ |(1 / 2 : ℝ)| < 1 := by
  norm_num [riceCounterSliceKernel, riceCounterDiskDomain]

/-- A positive measurable test function realizes the hypotheses of both actual expectation identities simultaneously.
Source: arXiv:2412.09080v3, §2.1, a nonzero probability-normalization example for the actual joint law. -/
example : Measurable (fun _ : ℝ × ℝ => (1 : ℝ≥0∞)) ∧
    (∫⁻ _, (1 : ℝ≥0∞) ∂riceCounterLaw) = ENNReal.ofReal Real.pi⁻¹ *
      ∫⁻ u, ∫⁻ y, ∫⁻ x, riceCounterSliceKernel u x y * 1 := by
  exact ⟨measurable_const, lintegral_riceCounterJointBase_kernel _ measurable_const⟩

end Transformer.Modes
