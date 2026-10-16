/-
# The actual probability regularity of the oscillating field

Source: arXiv:2412.09080v3, §2.1, hypotheses one and four
of `thm:kac-rice`. The explicit signed-scale disk field is
smooth for every sample. Under its actual probability law,
all field values are bounded by three and all actual time
derivatives by one, so both have finite second moments at
every observation time.

The already proved global derivative-modulus estimate shows
that a modulus larger than `ε` forces `|r|<η/ε` on the
almost-sure disk support. The original signed-scale law
assigns that interval probability at most `η/ε`. This gives
the exact printed `O(η)` condition, for every `ε>0`, with
constant `ε⁻¹`.

The probability estimate uses almost-everywhere set inclusion
and the actual coordinate pushforward, so it does not presume
measurability of an uncountable supremum. The original formal
condition uses the measure of that precise event, and the
estimate applies to it directly.
-/

import Transformer.Modes.Section2_MarginalDensity

open Real Set Filter MeasureTheory Topology Asymptotics
open scoped ENNReal

namespace Transformer.Modes

/-- Every actual field value and actual derivative are uniformly bounded for almost every original random parameter.
Source: arXiv:2412.09080v3, §2.1, the actual finite-moment bounds in hypothesis one of `thm:kac-rice`. -/
theorem ae_riceCounter_bounded : ∀ᵐ ω ∂riceCounterLaw, ∀ t : ℝ,
    |riceCounterField ω t| ≤ 3 ∧ |deriv (riceCounterField ω) t| ≤ 1 := by
  filter_upwards [ae_riceCounter_good] with ω hω t
  constructor
  · calc
      _ ≤ |ω.1| * (2 + ‖ω.2‖) := abs_riceCounterField_le ω t
      _ ≤ 1 * (2 + 1) := by
        gcongr
        · exact hω.2.1.le
        · exact hω.2.2.le
      _ = _ := by norm_num
  · rw [deriv_riceCounterField]
    exact (abs_riceCounterDerivative_le ω t).trans hω.2.2.le

/-- The actual field has a finite second moment at every real observation time under the original probability measure.
Source: arXiv:2412.09080v3, §2.1, the field-value finite-variance condition of hypothesis one. -/
theorem memLp_riceCounterField (t : ℝ) : MemLp (fun ω => riceCounterField ω t) 2 riceCounterLaw := by
  apply MemLp.of_bound (measurable_riceCounterField t).aestronglyMeasurable 3
  filter_upwards [ae_riceCounter_bounded] with ω hω
  simpa only [Real.norm_eq_abs] using (hω t).1

/-- The actual time derivative has a finite second moment at every real observation time under the original probability measure.
Source: arXiv:2412.09080v3, §2.1, the actual derivative finite-variance condition of hypothesis one. -/
theorem memLp_deriv_riceCounterField (t : ℝ) :
    MemLp (fun ω => deriv (riceCounterField ω) t) 2 riceCounterLaw := by
  apply MemLp.of_bound (measurable_deriv_riceCounterField t).aestronglyMeasurable 1
  filter_upwards [ae_riceCounter_bounded] with ω hω
  simpa only [Real.norm_eq_abs] using (hω t).2

/-- The actual sampled field is continuously differentiable on the entire real time line almost surely.
Source: arXiv:2412.09080v3, §2.1, the exact path-regularity condition of hypothesis one of `thm:kac-rice`. -/
theorem ae_contDiff_riceCounterField : ∀ᵐ ω ∂riceCounterLaw, ContDiff ℝ 1 (riceCounterField ω) :=
  ae_of_all _ (fun ω => (contDiff_riceCounterField ω).of_le (by simp))

/-- Under the actual product law, the small-scale event has its original signed-scale probability bound.
Source: arXiv:2412.09080v3, §2.1, the actual initialization probability used for hypothesis four of `thm:kac-rice`. -/
theorem riceCounter_small_scale_probability (a : ℝ) :
    riceCounterLaw {ω | |ω.1| < a} ≤ ENNReal.ofReal a := by
  have hs : MeasurableSet {r : ℝ | |r| < a} := (isOpen_lt continuous_abs continuous_const).measurableSet
  have h := congrArg (fun μ : Measure ℝ => μ {r : ℝ | |r| < a})
    (measurePreserving_fst (μ := riceCounterScaleLaw) (ν := riceCounterDiskLaw)).map_eq
  rw [Measure.map_apply measurable_fst hs] at h
  exact h.trans_le (riceCounterScale_small_mass_le a)

/-- The actual global derivative-modulus event has probability at most the exact ratio `η/ε`.
Source: arXiv:2412.09080v3, §2.1, the quantitative printed modulus condition in hypothesis four of `thm:kac-rice`. -/
theorem riceCounter_modulus_probability_le (ε η : ℝ) (hε : 0 < ε) (hη : 0 < η) :
    riceCounterLaw {ω | ENNReal.ofReal ε < modulusOfContinuity (deriv (riceCounterField ω)) η} ≤
      ENNReal.ofReal (η / ε) := by
  have hsub : {ω | ENNReal.ofReal ε < modulusOfContinuity (deriv (riceCounterField ω)) η} ≤ᵐ[riceCounterLaw]
      {ω | |ω.1| < η / ε} := by
    filter_upwards [ae_riceCounter_good] with ω hω hlarge
    exact riceCounter_large_modulus_imp_small_scale ω hω.2.2.le ε η hε hη hlarge
  exact (measure_mono_ae hsub).trans (riceCounter_small_scale_probability (η / ε))

/-- The real probability of the actual derivative-modulus event obeys the exact same positive-mesh ratio bound.
Source: arXiv:2412.09080v3, §2.1, the real probability used in hypothesis four of `thm:kac-rice`. -/
theorem riceCounter_modulus_probability_toReal_le (ε η : ℝ) (hε : 0 < ε) (hη : 0 < η) :
    (riceCounterLaw {ω | ENNReal.ofReal ε < modulusOfContinuity (deriv (riceCounterField ω)) η}).toReal ≤ η / ε := by
  exact (ENNReal.toReal_mono ENNReal.ofReal_ne_top (riceCounter_modulus_probability_le ε η hε hη)).trans_eq
    (ENNReal.toReal_ofReal (div_pos hη hε).le)

/-- The actual field satisfies exactly the printed `O(η)` derivative-modulus probability condition for every positive tolerance.
Source: arXiv:2412.09080v3, §2.1, the original hypothesis four of `thm:kac-rice`, retaining `O` rather than strengthening it. -/
theorem riceCounter_modulus_bigO (ε : ℝ) (hε : 0 < ε) :
    (fun η : ℝ => (riceCounterLaw {ω | ENNReal.ofReal ε <
      modulusOfContinuity (deriv (riceCounterField ω)) η}).toReal) =O[𝓝[>] (0 : ℝ)] fun η : ℝ => η := by
  apply IsBigO.of_bound ε⁻¹
  filter_upwards [self_mem_nhdsWithin (a := (0 : ℝ)) (s := Ioi 0)] with η hη
  change 0 < η at hη
  simp only [Real.norm_eq_abs, abs_of_nonneg ENNReal.toReal_nonneg, abs_of_pos hη]
  calc
    _ ≤ η / ε := riceCounter_modulus_probability_toReal_le ε η hε hη
    _ = ε⁻¹ * η := by rw [div_eq_mul_inv, mul_comm]

/-- The actual field, actual probability law, and derived actual densities jointly satisfy every original Kac–Rice hypothesis, including the printed `O(η)` condition.
Source: arXiv:2412.09080v3, §2.1, the complete unchanged hypothesis list of `thm:kac-rice`. -/
theorem isKacRiceField_riceCounter (T : Set ℝ) :
    IsKacRiceField riceCounterLaw riceCounterField 0 T
      (fun _ x => riceCounterValueDensity x) (fun _ x y => riceCounterJointDensity (x, y)) where
  contDiff := ae_contDiff_riceCounterField
  memLp := fun t _ => memLp_riceCounterField t
  memLp_deriv := fun t _ => memLp_deriv_riceCounterField t
  map_eq := fun t _ => map_riceCounterField_withDensity t
  continuousOn := ⟨Ioo (-1 : ℝ) 1, Ioo_mem_nhds (by norm_num) (by norm_num), continuousOn_riceCounterValueDensity T⟩
  map_joint_eq := fun t _ => map_riceCounter_joint_withDensity t
  continuousOn_joint := ⟨Ioo (-1 : ℝ) 1, Ioo_mem_nhds (by norm_num) (by norm_num), continuousOn_riceCounterJointDensity T⟩
  modulus := riceCounter_modulus_bigO

/-- Positive tolerance and mesh jointly realize every quantitative probability-bound hypothesis under the actual law.
Source: arXiv:2412.09080v3, §2.1, a genuine modulus-probability example for the actual field. -/
example : (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 4 ∧
    riceCounterLaw {ω | ENNReal.ofReal (1 / 2 : ℝ) <
      modulusOfContinuity (deriv (riceCounterField ω)) (1 / 4)} ≤ ENNReal.ofReal (1 / 2 : ℝ) := by
  refine ⟨by norm_num, by norm_num, ?_⟩
  convert riceCounter_modulus_probability_le (1 / 2) (1 / 4) (by norm_num) (by norm_num) using 1; norm_num

/-- The same positive tolerance realizes the full original asymptotic modulus hypothesis.
Source: arXiv:2412.09080v3, §2.1, a simultaneous positive-tolerance witness for hypothesis four of `thm:kac-rice`. -/
example : (0 : ℝ) < 1 / 2 ∧
    (fun η : ℝ => (riceCounterLaw {ω | ENNReal.ofReal (1 / 2 : ℝ) <
      modulusOfContinuity (deriv (riceCounterField ω)) η}).toReal) =O[𝓝[>] (0 : ℝ)] fun η : ℝ => η :=
  ⟨by norm_num, riceCounter_modulus_bigO _ (by norm_num)⟩

/-- A compact interval of positive length, an actual probability law, and the actual field and densities jointly realize every original hypothesis.
Source: arXiv:2412.09080v3, §2.1, a genuine simultaneous witness for all of `thm:kac-rice`'s premises. -/
example : IsProbabilityMeasure riceCounterLaw ∧ IsCompact (Icc (0 : ℝ) 1) ∧
    volume (Icc (0 : ℝ) 1) = 1 ∧
    IsKacRiceField riceCounterLaw riceCounterField 0 (Icc 0 1)
      (fun _ x => riceCounterValueDensity x) (fun _ x y => riceCounterJointDensity (x, y)) :=
  ⟨inferInstance, isCompact_Icc, by norm_num, isKacRiceField_riceCounter _⟩

/-- Both actual observations at time zero have finite second moments under the same original probability law.
Source: arXiv:2412.09080v3, §2.1, a simultaneous finite-variance example for hypothesis one. -/
example : MemLp (fun ω => riceCounterField ω 0) 2 riceCounterLaw ∧
    MemLp (fun ω => deriv (riceCounterField ω) 0) 2 riceCounterLaw :=
  ⟨memLp_riceCounterField 0, memLp_deriv_riceCounterField 0⟩

end Transformer.Modes
