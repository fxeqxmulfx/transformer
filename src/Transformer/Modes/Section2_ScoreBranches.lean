/-
# Three original score branches for Gaussian KDE critical-point charts

ArXiv:2412.09080v3, §2.1, `eq:Fn` and its derivative.
At a positive original bandwidth the score s exp(-βs²/2)
is strictly increasing between its two critical displacements
and strictly decreasing on each exterior interval. Each of these
three open branches has nonvanishing derivative and is injective.
They cover every displacement with nonzero original score derivative.

The critical radius is exactly 1/sqrt(β), not an assumed scale.
The sign estimates retain the original Gaussian exponential and
bandwidth factor. These branch facts will provide inverses for
solving the original field equation in one sample coordinate;
no inverse, implicit chart, or query nondegeneracy is postulated here.
-/

import Transformer.Modes.Section2_ParameterTransversality

open Real Set

namespace Transformer.Modes

/-- The two original critical score displacements have magnitude 1/sqrt(β).
Source: arXiv:2412.09080v3, §2.1, the derivative following `eq:Fn`. -/
noncomputable def kdeScoreRadius (β : ℝ) : ℝ := 1 / Real.sqrt β

/-- The original score is smooth in its displacement for every real bandwidth.
Source: arXiv:2412.09080v3, §2.1, the summands of `eq:Fn`. -/
theorem contDiff_kdeScore {k : WithTop ℕ∞} (β : ℝ) : ContDiff ℝ k (kdeScore β) := by
  unfold kdeScore
  fun_prop

/-- Positive original bandwidth gives a positive critical displacement radius.
Source: arXiv:2412.09080v3, §2.1, the critical factors of `eq:Fn`. -/
theorem kdeScoreRadius_pos {β : ℝ} (hβ : 0 < β) : 0 < kdeScoreRadius β :=
  one_div_pos.mpr (Real.sqrt_pos.mpr hβ)

/-- The original critical radius exactly cancels the bandwidth in the squared derivative factor.
Source: arXiv:2412.09080v3, §2.1, the derivative of `eq:Fn`. -/
theorem mul_kdeScoreRadius_sq {β : ℝ} (hβ : 0 < β) : β * kdeScoreRadius β ^ 2 = 1 := by
  unfold kdeScoreRadius
  rw [div_pow, one_pow, Real.sq_sqrt hβ.le]
  field_simp [hβ.ne']

/-- The original score derivative is positive throughout its middle open branch.
Source: arXiv:2412.09080v3, §2.1, the positive derivative factor in `eq:Fn`. -/
theorem kdeScore_deriv_pos_middle {β : ℝ} (hβ : 0 < β) (s : ℝ)
    (hs : s ∈ Ioo (-kdeScoreRadius β) (kdeScoreRadius β)) : 0 < deriv (kdeScore β) s := by
  have hr := kdeScoreRadius_pos hβ
  have hc := mul_kdeScoreRadius_sq hβ
  have hp : 0 < (kdeScoreRadius β - s) * (kdeScoreRadius β + s) :=
    mul_pos (by linarith [hs.2]) (by linarith [hs.1])
  have hsq : s ^ 2 < kdeScoreRadius β ^ 2 := by nlinarith
  have hm := mul_lt_mul_of_pos_left hsq hβ
  rw [(hasDerivAt_kdeScore β s).deriv]
  exact mul_pos (by linarith) (Real.exp_pos _)

/-- The original score derivative is negative on its right exterior branch.
Source: arXiv:2412.09080v3, §2.1, the negative derivative factor in `eq:Fn`. -/
theorem kdeScore_deriv_neg_right {β : ℝ} (hβ : 0 < β) (s : ℝ)
    (hs : s ∈ Ioi (kdeScoreRadius β)) : deriv (kdeScore β) s < 0 := by
  change kdeScoreRadius β < s at hs
  have hr := kdeScoreRadius_pos hβ
  have hc := mul_kdeScoreRadius_sq hβ
  have hp : 0 < (s - kdeScoreRadius β) * (s + kdeScoreRadius β) :=
    mul_pos (by linarith [hs]) (by linarith [hs])
  have hsq : kdeScoreRadius β ^ 2 < s ^ 2 := by nlinarith
  have hm := mul_lt_mul_of_pos_left hsq hβ
  rw [(hasDerivAt_kdeScore β s).deriv]
  exact mul_neg_of_neg_of_pos (by linarith) (Real.exp_pos _)

/-- The original score derivative is negative on its left exterior branch.
Source: arXiv:2412.09080v3, §2.1, the negative derivative factor in `eq:Fn`. -/
theorem kdeScore_deriv_neg_left {β : ℝ} (hβ : 0 < β) (s : ℝ)
    (hs : s ∈ Iio (-kdeScoreRadius β)) : deriv (kdeScore β) s < 0 := by
  change s < -kdeScoreRadius β at hs
  have hr := kdeScoreRadius_pos hβ
  have hc := mul_kdeScoreRadius_sq hβ
  have hp : 0 < (-s - kdeScoreRadius β) * (-s + kdeScoreRadius β) :=
    mul_pos (by linarith [hs]) (by linarith [hs])
  have hsq : kdeScoreRadius β ^ 2 < s ^ 2 := by nlinarith
  have hm := mul_lt_mul_of_pos_left hsq hβ
  rw [(hasDerivAt_kdeScore β s).deriv]
  exact mul_neg_of_neg_of_pos (by linarith) (Real.exp_pos _)

/-- The middle original score branch is strictly increasing.
Source: arXiv:2412.09080v3, §2.1, the branch structure of the derivative in `eq:Fn`. -/
theorem strictMonoOn_kdeScore_middle {β : ℝ} (hβ : 0 < β) :
    StrictMonoOn (kdeScore β) (Ioo (-kdeScoreRadius β) (kdeScoreRadius β)) := by
  apply strictMonoOn_of_deriv_pos (convex_Ioo _ _) (contDiff_kdeScore (k := 1) β).continuous.continuousOn
  rw [isOpen_Ioo.interior_eq]
  exact kdeScore_deriv_pos_middle hβ

/-- The right original score branch is strictly decreasing.
Source: arXiv:2412.09080v3, §2.1, the branch structure of the derivative in `eq:Fn`. -/
theorem strictAntiOn_kdeScore_right {β : ℝ} (hβ : 0 < β) :
    StrictAntiOn (kdeScore β) (Ioi (kdeScoreRadius β)) := by
  apply strictAntiOn_of_deriv_neg (convex_Ioi _) (contDiff_kdeScore (k := 1) β).continuous.continuousOn
  rw [isOpen_Ioi.interior_eq]
  exact kdeScore_deriv_neg_right hβ

/-- The left original score branch is strictly decreasing.
Source: arXiv:2412.09080v3, §2.1, the branch structure of the derivative in `eq:Fn`. -/
theorem strictAntiOn_kdeScore_left {β : ℝ} (hβ : 0 < β) :
    StrictAntiOn (kdeScore β) (Iio (-kdeScoreRadius β)) := by
  apply strictAntiOn_of_deriv_neg (convex_Iio _) (contDiff_kdeScore (k := 1) β).continuous.continuousOn
  rw [isOpen_Iio.interior_eq]
  exact kdeScore_deriv_neg_left hβ

/-- Each of the three original score branches is injective, so its inverse is unambiguous.
Source: arXiv:2412.09080v3, §2.1, the inverse branches of the summand of `eq:Fn`.
The separate branches retain the two actual critical displacements. -/
theorem kdeScore_branch_injective {β : ℝ} (hβ : 0 < β) :
    InjOn (kdeScore β) (Iio (-kdeScoreRadius β)) ∧
    InjOn (kdeScore β) (Ioo (-kdeScoreRadius β) (kdeScoreRadius β)) ∧
    InjOn (kdeScore β) (Ioi (kdeScoreRadius β)) := by
  exact ⟨(strictAntiOn_kdeScore_left hβ).injOn, (strictMonoOn_kdeScore_middle hβ).injOn,
    (strictAntiOn_kdeScore_right hβ).injOn⟩

/-- The three open original branches cover every displacement with nonzero score derivative.
Source: arXiv:2412.09080v3, §2.1, the only critical displacements in `eq:Fn`. -/
theorem kdeScore_regular_branch_cover {β : ℝ} (hβ : 0 < β) (s : ℝ)
    (hs : deriv (kdeScore β) s ≠ 0) :
    s ∈ Iio (-kdeScoreRadius β) ∨ s ∈ Ioo (-kdeScoreRadius β) (kdeScoreRadius β) ∨
      s ∈ Ioi (kdeScoreRadius β) := by
  by_cases hl : s < -kdeScoreRadius β
  · exact Or.inl hl
  by_cases hr : kdeScoreRadius β < s
  · exact Or.inr (Or.inr hr)
  right; left
  have hc := mul_kdeScoreRadius_sq hβ
  have hneL : s ≠ -kdeScoreRadius β := by
    intro he
    apply hs
    rw [he, (hasDerivAt_kdeScore β _).deriv]
    simp only [neg_sq, hc, sub_self, zero_mul]
  have hneR : s ≠ kdeScoreRadius β := by
    intro he
    apply hs
    rw [he, (hasDerivAt_kdeScore β _).deriv]
    simp only [hc, sub_self, zero_mul]
  constructor
  · by_contra h
    exact hneL (by linarith)
  · by_contra h
    exact hneR (by linarith)

/-- One actual positive bandwidth realizes all three nonempty original branches,
with a nonzero derivative at the middle point and both exterior points. -/
example : (0 : ℝ) < 1 ∧
    (0 : ℝ) ∈ Ioo (-kdeScoreRadius 1) (kdeScoreRadius 1) ∧
    (-2 : ℝ) ∈ Iio (-kdeScoreRadius 1) ∧ (2 : ℝ) ∈ Ioi (kdeScoreRadius 1) ∧
    deriv (kdeScore 1) 0 ≠ 0 := by
  norm_num [kdeScoreRadius, (hasDerivAt_kdeScore 1 0).deriv]

end Transformer.Modes
