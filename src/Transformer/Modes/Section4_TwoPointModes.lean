import Transformer.Modes.Section4_InteriorMode

/-
# The number of modes of a Gaussian KDE — two modes beyond a single sample

The claim `lem:scale-space`, arXiv:2412.09080v3, §4.2, bounds the number of
modes in `(a,∞)` by the number of sample points in `[a,∞)`, for every `a>0`.
Its proposed proof adds one Gaussian component at a time. A bound on the
total number of modes does not control modes crossing a fixed threshold.

Consider two samples, at `0` and `2`, with parameter `β=2`. Their KDE is a
positive constant times `exp(-t²) + exp(-(t-2)²)`. Its field `F_n` is
negative at zero and positive at `1/2`, since `exp(2)>3`. The compact-interval
maximum test therefore gives a genuine mode `t` with `0<t<1/2`.
Reflection about one gives another genuine mode at `2-t`.

At the positive threshold `a=t/2`, both modes are in `(a,∞)`. Among the
two sample points only `2` is in `[a,∞)`. The conclusion here gives an
explicit sample and bandwidth and an existential positive threshold;
it does not rely on a numerical approximation to either mode.

Source: arXiv:2412.09080v3, §1.1, `eq:gkde`; §2.1, `eq:Fn`;
§4.2, `lem:scale-space` and its proof.
-/

open Real
open scoped ENNReal

namespace Transformer.Modes

open Classical in
/-- The number of sample points in `S`, as used in the proposed pathwise
tail bound. Source: arXiv:2412.09080v3, §4.2, `lem:scale-space`. -/
noncomputable def countIn {n : ℕ} (X : Idx n → ℝ) (S : Set ℝ) : ℕ :=
  (Finset.univ.filter fun i => X i ∈ S).card

/-- The two fixed samples `0` and `2` used to refute the pathwise tail bound.
Source: arXiv:2412.09080v3, §4.2, counterexample to `lem:scale-space`. -/
def twoPointSample (i : Fin 2) : ℝ := 2 * (i.val : ℝ)

/-- The first center is zero.
Source: arXiv:2412.09080v3, §4.2, counterexample to `lem:scale-space`. -/
theorem twoPointSample_zero : twoPointSample 0 = 0 := by norm_num [twoPointSample]

/-- The second center is two.
Source: arXiv:2412.09080v3, §4.2, counterexample to `lem:scale-space`. -/
theorem twoPointSample_one : twoPointSample 1 = 2 := by norm_num [twoPointSample]

/-- The actual KDE, including its positive normalization, at the two samples.
Source: arXiv:2412.09080v3, §1.1, `eq:gkde`. -/
theorem kde_twoPointSample (β t : ℝ) :
    kde β twoPointSample t = Real.sqrt β / (2 * Real.sqrt (2 * π)) *
      (Real.exp (-(β / 2) * t ^ 2) + Real.exp (-(β / 2) * (t - 2) ^ 2)) := by
  simp only [kde, Fin.sum_univ_two, twoPointSample_zero, twoPointSample_one, sub_zero]
  norm_num

/-- The field in `eq:Fn` at `β=2` is a two-term finite sum.
Source: arXiv:2412.09080v3, §2.1, `eq:Fn`. -/
theorem fieldF_twoPointSample (t : ℝ) :
    fieldF 2 twoPointSample t = (1 / Real.sqrt 2) *
      (t * Real.exp (-(t ^ 2)) + (t - 2) * Real.exp (-((t - 2) ^ 2))) := by
  simp only [fieldF, Fin.sum_univ_two, twoPointSample_zero, twoPointSample_one, sub_zero]
  norm_num

/-- The contribution of the other sample pushes the derivative upwards at
the first sample, so the field is strictly negative there.
Source: arXiv:2412.09080v3, §2.1, `eq:Fn`; §4.2, `lem:scale-space`. -/
theorem fieldF_twoPointSample_zero_neg : fieldF 2 twoPointSample 0 < 0 := by
  rw [fieldF_twoPointSample]
  norm_num
  positivity

/-- At `1/2`, `exp(2)>3` reverses the field sign; this is an exact exponential
inequality rather than an approximate peak location.
Source: arXiv:2412.09080v3, §2.1, `eq:Fn`; §4.2, `lem:scale-space`. -/
theorem fieldF_twoPointSample_half_pos : 0 < fieldF 2 twoPointSample (1 / 2) := by
  have hval : fieldF 2 twoPointSample (1 / 2) = (1 / Real.sqrt 2) *
      ((1 / 2) * Real.exp (-1 / 4) - (3 / 2) * Real.exp (-9 / 4)) := by
    rw [fieldF_twoPointSample]
    norm_num
    ring
  have he : Real.exp (-1 / 4) = Real.exp 2 * Real.exp (-9 / 4) := by
    rw [← Real.exp_add]
    congr 1
    norm_num
  have h3 : (3 : ℝ) < Real.exp 2 := by
    linarith [Real.add_one_lt_exp (by norm_num : (2 : ℝ) ≠ 0)]
  rw [hval]
  apply mul_pos (by positivity)
  rw [he]
  nlinarith [Real.exp_pos (-9 / 4)]

/-- A left mode lies strictly to the right of the first sample at zero.
Source: arXiv:2412.09080v3, §4.2, counterexample to `lem:scale-space`. -/
theorem exists_left_mode_twoPointSample :
    ∃ t ∈ Set.Ioo (0 : ℝ) (1 / 2), IsLocalMax (kde 2 twoPointSample) t := by
  exact exists_isLocalMax_kde_Ioo_of_field_sign (by norm_num) (by norm_num)
    twoPointSample (by norm_num) fieldF_twoPointSample_zero_neg fieldF_twoPointSample_half_pos

/-- Every hypothesis of the interval maximum test holds at the chosen
sample, bandwidth, and endpoints. -/
example : (0 : ℝ) < 2 ∧ 0 < (2 : ℕ) ∧ (0 : ℝ) < 1 / 2 ∧
    fieldF 2 twoPointSample 0 < 0 ∧ 0 < fieldF 2 twoPointSample (1 / 2) := by
  exact ⟨by norm_num, by norm_num, by norm_num,
    fieldF_twoPointSample_zero_neg, fieldF_twoPointSample_half_pos⟩

/-- Reflection about the midpoint swaps the two centers and preserves their
KDE, for every real parameter. Source: arXiv:2412.09080v3, §1.1, `eq:gkde`. -/
theorem kde_twoPointSample_reflect (β t : ℝ) :
    kde β twoPointSample (2 - t) = kde β twoPointSample t := by
  rw [kde_twoPointSample, kde_twoPointSample]
  have h₀ : (2 - t) ^ 2 = (t - 2) ^ 2 := by ring
  have h₂ : (2 - t - 2) ^ 2 = t ^ 2 := by ring
  rw [h₀, h₂, add_comm]

/-- The two local maxima are distinct: one is in `(0,1/2)`, and its reflected
partner is in `(3/2,2)`.
Source: arXiv:2412.09080v3, §4.2, counterexample to `lem:scale-space`. -/
theorem exists_two_modes_twoPointSample :
    ∃ t ∈ Set.Ioo (0 : ℝ) (1 / 2), IsLocalMax (kde 2 twoPointSample) t ∧
      IsLocalMax (kde 2 twoPointSample) (2 - t) := by
  obtain ⟨t, ht, hmode⟩ := exists_left_mode_twoPointSample
  refine ⟨t, ht, hmode, ?_⟩
  have h' : IsLocalMax (kde 2 twoPointSample) (2 - (2 - t)) := by
    simpa only [sub_sub_cancel] using hmode
  have hc := h'.comp_continuous (g := fun x : ℝ => 2 - x) (b := 2 - t)
    (continuous_sub_left 2).continuousAt
  simpa only [Function.comp_def, kde_twoPointSample_reflect] using hc

/-- At a positive threshold below the shifted left mode, the actual tail
contains at least two modes even though the first sample is at zero.
Source: arXiv:2412.09080v3, §4.2, counterexample to `lem:scale-space`. -/
theorem exists_twoPointSample_tail_modes :
    ∃ a : ℝ, 0 < a ∧ a < 2 ∧ (2 : ℝ≥0∞) ≤ modeCount (kde 2 twoPointSample) (Set.Ioi a) := by
  obtain ⟨t, ht, hleft, hright⟩ := exists_two_modes_twoPointSample
  let a : ℝ := t / 2
  have hpos : 0 < a := by dsimp [a]; linarith [ht.1]
  have ha2 : a < 2 := by dsimp [a]; linarith [ht.2]
  refine ⟨a, hpos, ha2, ?_⟩
  have hne : t ≠ 2 - t := by intro h; linarith [ht.2]
  have hsub : ({t, 2 - t} : Set ℝ) ⊆ modeSet (kde 2 twoPointSample) (Set.Ioi a) := by
    rintro s (rfl | rfl)
    · exact ⟨Set.mem_Ioi.mpr (by dsimp [a]; linarith [ht.1]), hleft⟩
    · exact ⟨Set.mem_Ioi.mpr (by dsimp [a]; linarith [ht.2]), hright⟩
  have hcard := Set.encard_le_encard hsub
  rw [Set.encard_pair hne] at hcard
  simpa only [modeCount, ENat.toENNReal_ofNat] using ENat.toENNReal_le.mpr hcard

/-- At every threshold between the centers, exactly one sample lies on the
right. Source: arXiv:2412.09080v3, §4.2, counterexample to `lem:scale-space`. -/
theorem countIn_twoPointSample {a : ℝ} (ha : 0 < a) (ha2 : a ≤ 2) :
    countIn twoPointSample (Set.Ici a) = 1 := by
  classical
  unfold countIn
  rw [Finset.univ_fin2]
  simp [Finset.filter_insert, Finset.filter_singleton, twoPointSample_zero,
    twoPointSample_one, ha.not_ge, ha2]

/-- The sample-count hypotheses have the concrete threshold `a=1`. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 2 := ⟨one_pos, by norm_num⟩

end Transformer.Modes
