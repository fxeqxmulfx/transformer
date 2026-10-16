/-
# Real coordinates for the actual disk law

Source: arXiv:2412.09080v3, §2.1, the actual joint-law
hypothesis in `thm:kac-rice`. The oscillating field uses a
uniform signed scale and an independent uniform disk point.
Its time-independent joint law is described in real disk
coordinates `(re z, -im z)`.

Conjugation followed by the standard complex-to-real-product
equivalence preserves Lebesgue volume. Restricting to the
actual open unit disk and retaining its normalization gives
the exact disk-coordinate law. The support is the ordinary
Euclidean disk `u²+y²<1`, with both coordinate magnitudes
strictly below one.

A positive stretch of the original signed scale has its
literal normalized interval law. Both changes of coordinates
are stated as equalities of measures and nonnegative
integrals; no density for the final joint observation is
assumed. Their simultaneous examples use positive stretch
and an actual interior disk point.
-/

import Transformer.Modes.Section2_DiskRotations

open Real Set Filter MeasureTheory
open scoped ENNReal

namespace Transformer.Modes

/-- The actual real disk coordinates have the derivative sign of the time-zero joint observation.
Source: arXiv:2412.09080v3, §2.1, the disk coordinates for hypothesis three of `thm:kac-rice`. -/
def riceCounterDiskCoords (z : ℂ) : ℝ × ℝ := (z.re, -z.im)

/-- The actual disk-coordinate support is the Euclidean open unit disk.
Source: arXiv:2412.09080v3, §2.1, the integration domain of the explicit joint-law construction. -/
def riceCounterDiskDomain : Set (ℝ × ℝ) := {q | q.1 ^ 2 + q.2 ^ 2 < 1}

/-- The actual disk-coordinate map is measurable.
Source: arXiv:2412.09080v3, §2.1, the measurable coordinate substitution for `thm:kac-rice`. -/
theorem measurable_riceCounterDiskCoords : Measurable riceCounterDiskCoords :=
  Complex.reCLM.continuous.measurable.prodMk Complex.imCLM.continuous.measurable.neg

/-- The actual disk-coordinate domain is a measurable subset of the real product space.
Source: arXiv:2412.09080v3, §2.1, the domain of the joint-law integral for `thm:kac-rice`. -/
theorem measurableSet_riceCounterDiskDomain : MeasurableSet riceCounterDiskDomain := by
  exact (isOpen_lt (by fun_prop : Continuous (fun q : ℝ × ℝ => q.1 ^ 2 + q.2 ^ 2)) continuous_const).measurableSet

/-- The actual signed complex coordinates preserve real two-dimensional Lebesgue volume.
Source: arXiv:2412.09080v3, §2.1, the volume change in the joint-law calculation. -/
theorem measurePreserving_riceCounterDiskCoords :
    MeasurePreserving riceCounterDiskCoords (volume : Measure ℂ) (volume : Measure (ℝ × ℝ)) := by
  exact Complex.volume_preserving_equiv_real_prod.comp Complex.conjLIE.measurePreserving

/-- The actual coordinate preimage of the real disk is precisely the original complex disk.
Source: arXiv:2412.09080v3, §2.1, the original bounded support of the field construction. -/
theorem riceCounterDiskCoords_preimage_domain :
    riceCounterDiskCoords ⁻¹' riceCounterDiskDomain = Metric.ball (0 : ℂ) 1 := by
  ext z
  change z.re ^ 2 + (-z.im) ^ 2 < 1 ↔ dist z 0 < 1
  have hs : ‖z‖ ^ 2 = z.re ^ 2 + z.im ^ 2 := by rw [Complex.sq_norm, Complex.normSq_apply]; ring
  rw [dist_zero_right, neg_sq, ← hs]
  exact sq_lt_one_iff₀ (norm_nonneg z)

/-- Each actual disk coordinate has magnitude strictly below one, so the field-value stretch is strictly positive.
Source: arXiv:2412.09080v3, §2.1, the support bounds in the explicit joint-density calculation. -/
theorem riceCounterDiskDomain_bounds (q : ℝ × ℝ) (hq : q ∈ riceCounterDiskDomain) :
    |q.1| < 1 ∧ |q.2| < 1 ∧ 0 < 2 + q.1 := by
  have h : q.1 ^ 2 + q.2 ^ 2 < 1 := hq
  have hu : |q.1| < 1 := (sq_lt_one_iff_abs_lt_one q.1).mp (by nlinarith [sq_nonneg q.2])
  have hy : |q.2| < 1 := (sq_lt_one_iff_abs_lt_one q.2).mp (by nlinarith [sq_nonneg q.1])
  exact ⟨hu, hy, by linarith [(abs_lt.mp hu).1]⟩

/-- On the actual disk the field-value stretch lies strictly between one and three.
Source: arXiv:2412.09080v3, §2.1, the quantitative denominator bounds for the actual joint density. -/
theorem riceCounterDiskDomain_stretch_bounds (q : ℝ × ℝ) (hq : q ∈ riceCounterDiskDomain) :
    1 < 2 + q.1 ∧ 2 + q.1 < 3 := by
  have h := abs_lt.mp (riceCounterDiskDomain_bounds q hq).1
  constructor <;> linarith [h.1, h.2]

/-- Every field value of magnitude below one lies in the stretched scale interval for every actual disk point.
Source: arXiv:2412.09080v3, §2.1, the neighborhood of zero on which the joint density is independent of the field value. -/
theorem riceCounterDiskDomain_value_interval (q : ℝ × ℝ) (hq : q ∈ riceCounterDiskDomain)
    (x : ℝ) (hx : |x| < 1) : x ∈ Ioo (-(2 + q.1)) (2 + q.1) := by
  have hc := (riceCounterDiskDomain_stretch_bounds q hq).1
  have h := abs_lt.mp hx
  constructor <;> linarith [h.1, h.2]

/-- Swapping the two real coordinates leaves the actual disk-coordinate domain invariant.
Source: arXiv:2412.09080v3, §2.1, the disk-domain symmetry in the Tonelli calculation of the joint density. -/
theorem riceCounterDiskDomain_swap (q : ℝ × ℝ) :
    q.swap ∈ riceCounterDiskDomain ↔ q ∈ riceCounterDiskDomain := by
  change q.2 ^ 2 + q.1 ^ 2 < 1 ↔ q.1 ^ 2 + q.2 ^ 2 < 1
  rw [add_comm]

/-- The actual disk-coordinate law is its literal normalized Lebesgue restriction, including the original area constant.
Source: arXiv:2412.09080v3, §2.1, the real-coordinate law used to derive the joint density in `thm:kac-rice`. -/
theorem map_riceCounterDisk_coords :
    riceCounterDiskLaw.map riceCounterDiskCoords =
      ENNReal.ofReal Real.pi⁻¹ • (volume : Measure (ℝ × ℝ)).restrict riceCounterDiskDomain := by
  have h := measurePreserving_riceCounterDiskCoords.restrict_preimage measurableSet_riceCounterDiskDomain
  rw [riceCounterDiskCoords_preimage_domain] at h
  rw [riceCounterDiskLaw, Measure.map_smul _ measurable_riceCounterDiskCoords.aemeasurable, h.map_eq]

/-- Nonnegative expectations of the actual disk coordinates equal the normalized real-disk integral.
Source: arXiv:2412.09080v3, §2.1, the exact coordinate substitution in the actual joint-law calculation. -/
theorem lintegral_riceCounterDisk_coords (g : ℝ × ℝ → ℝ≥0∞) (hg : Measurable g) :
    ∫⁻ z, g (riceCounterDiskCoords z) ∂riceCounterDiskLaw =
      ENNReal.ofReal Real.pi⁻¹ * ∫⁻ q in riceCounterDiskDomain, g q := by
  rw [← lintegral_map hg measurable_riceCounterDiskCoords, map_riceCounterDisk_coords,
    lintegral_smul_measure, smul_eq_mul]

/-- A positive stretch of the original signed scale has the actual normalized stretched-interval law.
Source: arXiv:2412.09080v3, §2.1, the scalar change of variable in the explicit joint density. -/
theorem map_riceCounterScale_stretch (c : ℝ) (hc : 0 < c) :
    riceCounterScaleLaw.map (fun r => c * r) =
      ENNReal.ofReal (1 / (2 * c)) • (volume : Measure ℝ).restrict (Ioo (-c) c) := by
  have hm : Measurable (fun r : ℝ => c * r) := measurable_const.mul measurable_id
  have hpre : (fun r : ℝ => c * r) ⁻¹' Ioo (-c) c = Ioo (-1) 1 := by
    ext r
    change (-c < c * r ∧ c * r < c) ↔ (-1 < r ∧ r < 1)
    constructor
    · intro h; constructor <;> nlinarith [h.1, h.2]
    · intro h; constructor <;> nlinarith [h.1, h.2]
  have h := Measure.restrict_map (μ := (volume : Measure ℝ)) hm (s := Ioo (-c) c) measurableSet_Ioo
  rw [hpre, Real.map_volume_mul_left hc.ne', Measure.restrict_smul] at h
  rw [riceCounterScaleLaw, Measure.map_smul _ hm.aemeasurable, ← h, smul_smul]
  congr 1
  rw [← ENNReal.ofReal_mul (by norm_num : (0 : ℝ) ≤ 1 / 2), abs_of_pos (inv_pos.mpr hc)]
  congr 1
  field_simp

/-- The actual scale integral obeys the exact positive-stretch substitution with its Jacobian factor.
Source: arXiv:2412.09080v3, §2.1, the scalar substitution deriving hypothesis three's actual density. -/
theorem lintegral_riceCounterScale_stretch (c : ℝ) (hc : 0 < c)
    (g : ℝ → ℝ≥0∞) (hg : Measurable g) :
    ∫⁻ r, g (c * r) ∂riceCounterScaleLaw =
      ENNReal.ofReal (1 / (2 * c)) * ∫⁻ x in Ioo (-c) c, g x := by
  rw [← lintegral_map (g := fun r : ℝ => c * r) hg (measurable_const.mul measurable_id), map_riceCounterScale_stretch c hc,
    lintegral_smul_measure, smul_eq_mul]

/-- A positive stretch and a genuine interior disk point jointly realize all coordinate-substitution hypotheses.
Source: arXiv:2412.09080v3, §2.1, nondegenerate witnesses for the actual joint-law construction. -/
example : (0 : ℝ) < 2 ∧ ((1 / 2 : ℝ), (1 / 2 : ℝ)) ∈ riceCounterDiskDomain ∧
    riceCounterScaleLaw.map (fun r => (2 : ℝ) * r) =
      ENNReal.ofReal (1 / 4 : ℝ) • (volume : Measure ℝ).restrict (Ioo (-2) 2) ∧
    (∫⁻ _, (1 : ℝ≥0∞) ∂riceCounterScaleLaw) =
      ENNReal.ofReal (1 / 4 : ℝ) * ∫⁻ _ in Ioo (-2 : ℝ) 2, (1 : ℝ≥0∞) := by
  refine ⟨by norm_num, by norm_num [riceCounterDiskDomain], ?_, ?_⟩
  · convert map_riceCounterScale_stretch 2 (by norm_num) using 1; norm_num
  · convert lintegral_riceCounterScale_stretch 2 (by norm_num) (fun _ => 1) measurable_const using 1; norm_num

/-- The same genuine disk point realizes the denominator and near-zero interval bounds simultaneously.
Source: arXiv:2412.09080v3, §2.1, joint witnesses for the nontrivial neighborhood used in the actual density. -/
example : ((1 / 2 : ℝ), (1 / 2 : ℝ)) ∈ riceCounterDiskDomain ∧ |(1 / 2 : ℝ)| < 1 ∧
    1 < 2 + (1 / 2 : ℝ) ∧ 2 + (1 / 2 : ℝ) < 3 ∧
    (1 / 2 : ℝ) ∈ Ioo (-(2 + (1 / 2 : ℝ))) (2 + (1 / 2 : ℝ)) := by
  norm_num [riceCounterDiskDomain]

end Transformer.Modes
