module

public import Transformer.RoPE.Basic

/-!
# Field-constrained optimal frequency

Wu, Liu and Jadbabaie, *How Data Shapes RoPE Frequency Usage*,
arXiv:2607.07678v1, §3, Theorem 1. The probability distribution is genuinely
integrated; it may have atoms. Its nonzero-distance mass is essential for
strictness and uniqueness. This is optimality for positional contrast, not
for a transformer's prediction loss.
-/

@[expose] public section

open MeasureTheory Set

namespace Transformer.RoPE.FrequencyMatching

/-- Wu et al. §3.3: contrast is monotone on admissible frequencies throughout
the dependency window, and strictly monotone at every positive distance. -/
theorem contrast_mono {width a b distance : ℝ}
    (ha : Admissible width a) (hb : Admissible width b)
    (hab : a ≤ b) (hr : distance ∈ Icc 0 width) :
    contrast a distance ≤ contrast b distance := by
  have hphase (omega : ℝ) (h : Admissible width omega) :
      omega * distance ∈ Icc 0 Real.pi :=
    ⟨mul_nonneg h.1.le hr.1, (mul_le_mul_of_nonneg_left hr.2 h.1.le).trans h.2⟩
  have := Real.strictAntiOn_cos.antitoneOn (hphase a ha) (hphase b hb)
    (mul_le_mul_of_nonneg_right hab hr.1)
  unfold contrast
  linarith

example : contrast (Real.pi / 2) 1 ≤ contrast Real.pi 1 :=
  contrast_mono ⟨by positivity, by nlinarith [Real.pi_pos]⟩
    ⟨Real.pi_pos, by simp⟩ (by nlinarith [Real.pi_pos]) ⟨by norm_num, le_rfl⟩

/-- Wu et al. Theorem 1, strict pointwise comparison away from zero. -/
theorem contrast_strict {width a b distance : ℝ}
    (ha : Admissible width a) (hb : Admissible width b)
    (hab : a < b) (hr : distance ∈ Ioc 0 width) :
    contrast a distance < contrast b distance := by
  have hphase (omega : ℝ) (h : Admissible width omega) :
      omega * distance ∈ Icc 0 Real.pi :=
    ⟨mul_nonneg h.1.le hr.1.le, (mul_le_mul_of_nonneg_left hr.2 h.1.le).trans h.2⟩
  have := Real.strictAntiOn_cos (hphase a ha) (hphase b hb)
    (mul_lt_mul_of_pos_right hab hr.1)
  unfold contrast
  linarith

example : contrast (Real.pi / 2) 1 < contrast Real.pi 1 :=
  contrast_strict ⟨by positivity, by nlinarith [Real.pi_pos]⟩
    ⟨Real.pi_pos, by simp⟩ (by nlinarith [Real.pi_pos]) ⟨by norm_num, le_rfl⟩

/-- Wu et al. Theorem 1: the actual integral is strictly increasing over the
admissible set, assuming positive mass at nonzero distances. -/
theorem utility_strictMonoOn (mu : Measure ℝ) [IsProbabilityMeasure mu]
    {width : ℝ} (hwindow : ∀ᵐ r ∂mu, r ∈ Icc 0 width)
    (hmass : 0 < mu (Ioc 0 width)) :
    StrictMonoOn (utility mu) {omega | Admissible width omega} := by
  intro a ha b hb hab
  let gap : ℝ → ℝ := fun r => contrast b r - contrast a r
  have hint : Integrable gap mu := (contrast_integrable mu b).sub (contrast_integrable mu a)
  have hnonneg : 0 ≤ᵐ[mu] gap := hwindow.mono fun r hr =>
    sub_nonneg.mpr (contrast_mono ha hb hab.le hr)
  have hsupport : Ioc 0 width ⊆ Function.support gap := by
    intro r hr
    exact ne_of_gt (sub_pos.mpr (contrast_strict ha hb hab hr))
  have hpositive : 0 < ∫ r, gap r ∂mu :=
    (integral_pos_iff_support_of_nonneg_ae hnonneg hint).2
      (hmass.trans_le (measure_mono hsupport))
  rw [show (∫ r, gap r ∂mu) = utility mu b - utility mu a from
    integral_sub (contrast_integrable mu b) (contrast_integrable mu a)] at hpositive
  exact sub_pos.mp hpositive

example : StrictMonoOn (utility (Measure.dirac 1))
    {omega | Admissible 1 omega} :=
  utility_strictMonoOn (Measure.dirac 1) (by simp) (by simp)

/-- Wu et al. Theorem 1: `pi / width` uniquely maximizes the contrast utility.
All support and positive-mass assumptions from the source are retained. -/
theorem unique_optimal_frequency (mu : Measure ℝ) [IsProbabilityMeasure mu]
    {width : ℝ} (hwidth : 0 < width)
    (hwindow : ∀ᵐ r ∂mu, r ∈ Icc 0 width) (hmass : 0 < mu (Ioc 0 width)) :
    Admissible width (Real.pi / width) ∧
      ∀ omega, Admissible width omega →
        utility mu omega ≤ utility mu (Real.pi / width) ∧
          (utility mu omega = utility mu (Real.pi / width) ↔ omega = Real.pi / width) := by
  have hbest : Admissible width (Real.pi / width) :=
    ⟨div_pos Real.pi_pos hwidth, by rw [div_mul_cancel₀ _ hwidth.ne']⟩
  have hmono := utility_strictMonoOn mu hwindow hmass
  refine ⟨hbest, fun omega homega => ?_⟩
  have hle : omega ≤ Real.pi / width := (le_div_iff₀ hwidth).2 homega.2
  refine ⟨hmono.monotoneOn homega hbest hle, ?_⟩
  constructor
  · intro heq
    by_contra hne
    exact (ne_of_lt (hmono homega hbest (lt_of_le_of_ne hle hne))) heq
  · rintro rfl
    rfl

example : Admissible 1 (Real.pi / 1) ∧
    ∀ omega, Admissible 1 omega →
      utility (Measure.dirac 1) omega ≤ utility (Measure.dirac 1) (Real.pi / 1) ∧
        (utility (Measure.dirac 1) omega = utility (Measure.dirac 1) (Real.pi / 1) ↔
          omega = Real.pi / 1) :=
  unique_optimal_frequency (Measure.dirac 1) (by norm_num) (by simp) (by simp)

/-- The nonzero-mass hypothesis cannot be dropped: a dependency distribution
concentrated at zero has zero contrast at every frequency. Wu et al. Theorem 1. -/
theorem zero_distance_no_unique_optimum :
    utility (Measure.dirac 0) 1 = utility (Measure.dirac 0) 2 ∧ (1 : ℝ) ≠ 2 := by
  norm_num [utility, contrast]

end Transformer.RoPE.FrequencyMatching
