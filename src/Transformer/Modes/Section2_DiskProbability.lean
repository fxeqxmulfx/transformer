module

public import Transformer.Modes.Section2_OscillatingField
public import Mathlib.MeasureTheory.Measure.Lebesgue.VolumeOfBalls

/-!
# The signed scale and disk probability law

Source: arXiv:2412.09080v3, §2.1, the probability hypotheses
of `thm:kac-rice`. The oscillating field is sampled with a
uniform signed scale in `(-1,1)` and an independent uniform
point in the complex unit disk.

Both factors are literal normalized restrictions of Lebesgue
volume. Their product is a probability measure. Almost every
sample has a nonzero signed scale, absolute scale less than
one, and disk radius less than one. The scale-zero path is
therefore excluded by an actual null-set calculation.

The scale interval of radius `a` has probability at most `a`.
For `a ≤ 1` the bound is exact. This supplies the small-scale
probability estimate required by the original `O(η)` modulus
condition when the deterministic modulus estimate is applied.
The example uses a genuinely positive scale interval, so the
probability law and its small-scale events are nontrivial.
-/

@[expose] public section

open Real Set Filter MeasureTheory
open scoped ENNReal

namespace Transformer.Modes

/-- The signed scale is uniform on the original real interval `(-1,1)`.
Source: arXiv:2412.09080v3, §2.1, the scale law for the field testing hypothesis four of `thm:kac-rice`. -/
noncomputable def riceCounterScaleLaw : Measure ℝ :=
  ENNReal.ofReal (1 / 2 : ℝ) • (volume : Measure ℝ).restrict (Ioo (-1) 1)

/-- The disk parameter is uniform in the complex unit disk, with its actual area normalization.
Source: arXiv:2412.09080v3, §2.1, the disk law for the oscillating field testing `thm:kac-rice`. -/
noncomputable def riceCounterDiskLaw : Measure ℂ :=
  ENNReal.ofReal (Real.pi⁻¹) • (volume : Measure ℂ).restrict (Metric.ball 0 1)

/-- The actual scale and disk coordinates are independent under their product probability law.
Source: arXiv:2412.09080v3, §2.1, the initialization of the explicit random field testing `thm:kac-rice`. -/
noncomputable def riceCounterLaw : Measure (ℝ × ℂ) :=
  riceCounterScaleLaw.prod riceCounterDiskLaw

instance : IsProbabilityMeasure riceCounterScaleLaw := by
  constructor
  rw [riceCounterScaleLaw, Measure.smul_apply, Measure.restrict_apply_univ, Real.volume_Ioo, smul_eq_mul,
    ← ENNReal.ofReal_mul (by norm_num : (0 : ℝ) ≤ 1 / 2)]
  norm_num

instance : IsProbabilityMeasure riceCounterDiskLaw := by
  constructor
  rw [riceCounterDiskLaw, Measure.smul_apply, Measure.restrict_apply_univ, Complex.volume_ball, smul_eq_mul]
  simp only [ENNReal.ofReal_one, one_pow, one_mul, ENNReal.coe_nnreal_eq, NNReal.coe_real_pi]
  rw [← ENNReal.ofReal_mul (inv_nonneg.mpr Real.pi_pos.le), inv_mul_cancel₀ Real.pi_pos.ne', ENNReal.ofReal_one]

instance : IsProbabilityMeasure riceCounterLaw := by
  unfold riceCounterLaw
  infer_instance

/-- Almost every actual scale lies in the defining open signed interval.
Source: arXiv:2412.09080v3, §2.1, the actual finite-moment scale support used for `thm:kac-rice`. -/
theorem ae_riceCounterScale_mem : ∀ᵐ r ∂riceCounterScaleLaw, r ∈ Ioo (-1 : ℝ) 1 := by
  unfold riceCounterScaleLaw
  exact Measure.ae_smul_measure (ae_restrict_mem measurableSet_Ioo) _

/-- The actual scale-zero set has probability zero in the original real initialization coordinate.
Source: arXiv:2412.09080v3, §2.1, the exceptional constant path in the explicit field testing `thm:kac-rice`. -/
theorem riceCounterScale_singleton_zero : riceCounterScaleLaw ({0} : Set ℝ) = 0 := by
  norm_num [riceCounterScaleLaw, Measure.restrict_apply (measurableSet_singleton (0 : ℝ))]

/-- The actual sampled scale is nonzero almost surely.
Source: arXiv:2412.09080v3, §2.1, excluding the actual exceptional zero-scale path in the construction for `thm:kac-rice`. -/
theorem ae_riceCounterScale_ne_zero : ∀ᵐ r ∂riceCounterScaleLaw, r ≠ 0 := by
  rw [ae_iff]
  simpa only [not_not, ← mem_singleton_iff, ofPred_mem_eq] using riceCounterScale_singleton_zero

/-- Almost every actual disk parameter lies in its defining open unit disk.
Source: arXiv:2412.09080v3, §2.1, the actual bounded disk support in the construction for `thm:kac-rice`. -/
theorem ae_riceCounterDisk_mem : ∀ᵐ z ∂riceCounterDiskLaw, z ∈ Metric.ball (0 : ℂ) 1 := by
  unfold riceCounterDiskLaw
  exact Measure.ae_smul_measure (ae_restrict_mem measurableSet_ball) _

/-- The actual disk radius is strictly less than one almost surely.
Source: arXiv:2412.09080v3, §2.1, the field and derivative amplitude bounds required for `thm:kac-rice`. -/
theorem ae_riceCounterDisk_norm_lt_one : ∀ᵐ z ∂riceCounterDiskLaw, ‖z‖ < 1 := by
  filter_upwards [ae_riceCounterDisk_mem] with z hz
  simpa only [Metric.mem_ball, dist_zero_right] using hz

/-- The product probability law jointly realizes the actual nonzero signed scale and disk-radius conditions almost surely.
Source: arXiv:2412.09080v3, §2.1, the actual random parameters of the field testing `thm:kac-rice`. -/
theorem ae_riceCounter_good : ∀ᵐ ω ∂riceCounterLaw, ω.1 ≠ 0 ∧ |ω.1| < 1 ∧ ‖ω.2‖ < 1 := by
  have hR := (measurePreserving_fst (μ := riceCounterScaleLaw) (ν := riceCounterDiskLaw)).quasiMeasurePreserving
  have hZ := (measurePreserving_snd (μ := riceCounterScaleLaw) (ν := riceCounterDiskLaw)).quasiMeasurePreserving
  filter_upwards [hR.ae ae_riceCounterScale_ne_zero, hR.ae ae_riceCounterScale_mem,
    hZ.ae ae_riceCounterDisk_norm_lt_one] with ω hr hscale hz
  exact ⟨hr, abs_lt.mpr hscale, hz⟩

/-- The actual signed-scale interval of any radius has probability at most the nonnegative part of that radius.
Source: arXiv:2412.09080v3, §2.1, the exact small-scale probability control for the modulus condition in `thm:kac-rice`. -/
theorem riceCounterScale_small_mass_le (a : ℝ) :
    riceCounterScaleLaw {r : ℝ | |r| < a} ≤ ENNReal.ofReal a := by
  have he : {r : ℝ | |r| < a} = Ioo (-a) a := by
    apply Set.ext
    intro r
    change |r| < a ↔ -a < r ∧ r < a
    exact abs_lt
  rw [he, riceCounterScaleLaw, Measure.smul_apply, smul_eq_mul]
  calc
    _ ≤ ENNReal.ofReal (1 / 2 : ℝ) * volume (Ioo (-a) a) :=
      by gcongr; exact Measure.restrict_le_self
    _ = _ := by
      rw [Real.volume_Ioo, ← ENNReal.ofReal_mul (by norm_num : (0 : ℝ) ≤ 1 / 2)]
      congr 1
      ring

/-- For radii at most one, the actual small-scale interval probability is exactly its radius clipped at zero.
Source: arXiv:2412.09080v3, §2.1, the precise small-scale law for the oscillating field testing `thm:kac-rice`. -/
theorem riceCounterScale_small_mass_eq (a : ℝ) (ha : a ≤ 1) :
    riceCounterScaleLaw {r : ℝ | |r| < a} = ENNReal.ofReal a := by
  have he : {r : ℝ | |r| < a} = Ioo (-a) a := by
    apply Set.ext
    intro r
    change |r| < a ↔ -a < r ∧ r < a
    exact abs_lt
  have hsub : Ioo (-a) a ⊆ Ioo (-1 : ℝ) 1 := by intro r hr; constructor <;> linarith [hr.1, hr.2]
  rw [he, riceCounterScaleLaw, Measure.smul_apply, smul_eq_mul, Measure.restrict_apply measurableSet_Ioo,
    inter_eq_left.mpr hsub, Real.volume_Ioo, ← ENNReal.ofReal_mul (by norm_num : (0 : ℝ) ≤ 1 / 2)]
  congr 1
  ring

/-- The actual field is nonzero at every time for almost every sample from the actual product probability law.
Source: arXiv:2412.09080v3, §2.1, the level-zero behavior of the explicit random field testing `thm:kac-rice`. -/
theorem ae_riceCounterField_ne_zero : ∀ᵐ ω ∂riceCounterLaw, ∀ t : ℝ, riceCounterField ω t ≠ 0 := by
  filter_upwards [ae_riceCounter_good] with ω hω t
  exact riceCounterField_ne_zero ω hω.1 hω.2.2.le t

/-- The original expected zero-level upcrossing count is exactly zero on every time set for the actual product probability law.
Source: arXiv:2412.09080v3, §2.1, the original left side of the Kac–Rice formula for the oscillating construction. -/
theorem expectedUpcrossings_riceCounter_eq_zero (T : Set ℝ) :
    expectedUpcrossings riceCounterLaw riceCounterField 0 T = 0 := by
  unfold expectedUpcrossings
  have h : (fun ω => upcrossingCount (riceCounterField ω) 0 T) =ᵐ[riceCounterLaw] fun _ => 0 := by
    filter_upwards [ae_riceCounter_good] with ω hω
    exact riceCounterField_upcrossingCount_eq_zero ω hω.1 hω.2.2.le T
  rw [lintegral_congr_ae h]
  simp

/-- A genuine positive-radius interval has strictly positive probability under the actual scale law, and the product law is an actual probability measure.
Source: arXiv:2412.09080v3, §2.1, a nontrivial probability witness for the construction testing `thm:kac-rice`. -/
example : IsProbabilityMeasure riceCounterLaw ∧ (1 / 2 : ℝ) ≤ 1 ∧
    riceCounterScaleLaw {r : ℝ | |r| < 1 / 2} = ENNReal.ofReal (1 / 2 : ℝ) := by
  exact ⟨inferInstance, by norm_num, riceCounterScale_small_mass_eq _ (by norm_num)⟩

/-- The actual zero expected count holds on a compact time interval of positive Lebesgue measure.
Source: arXiv:2412.09080v3, §2.1, the nondegenerate time set in the construction testing `thm:kac-rice`. -/
example : IsCompact (Icc (0 : ℝ) 1) ∧ volume (Icc (0 : ℝ) 1) = 1 ∧
    expectedUpcrossings riceCounterLaw riceCounterField 0 (Icc 0 1) = 0 :=
  ⟨isCompact_Icc, by norm_num, expectedUpcrossings_riceCounter_eq_zero _⟩

end Transformer.Modes
