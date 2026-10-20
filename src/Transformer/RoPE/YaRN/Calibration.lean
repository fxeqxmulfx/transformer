module

public import Transformer.RoPE.YaRN.Retention

/-!
# Exact calibration and the limits of a length-only temperature

Derived from the explicit retention objective in `YaRN.Retention`.
This is not a derivation of YaRN's empirical constant as a universal law.
The square-root law is exact for an equal-tail model, and preserves a
worst-case mass bound for arbitrary logits with a certified positive gap.
-/

@[expose] public section

namespace Transformer.RoPE.YaRN

/-- The unique logit gain preserving two-level winner probability when the
number of competitors grows. `gap` remains an essential model quantity. -/
noncomputable def retentionGain (original extended gap : ℝ) : ℝ :=
  1 + Real.log (extended / original) / gap

/-- Exact retention fixes the gain uniquely for every nonzero gap. -/
theorem retention_gain_unique {original extended gap : ℝ}
    (ho : 0 < original) (he : 0 < extended) (hd : gap ≠ 0) (gain : ℝ) :
    twoLevelMass extended gap gain = twoLevelMass original gap 1 ↔
      gain = retentionGain original extended gap := by
  rw [mass_preservation_iff ho he]
  have hid : retentionGain original extended gap * gap =
      gap + Real.log (extended / original) := by
    unfold retentionGain
    field_simp
  constructor
  · intro h
    exact (mul_right_cancel₀ hd) (h.trans hid.symm)
  · rintro rfl
    exact hid

example : twoLevelMass 255 5 (retentionGain 63 255 5) = twoLevelMass 63 5 1 :=
  (retention_gain_unique (by norm_num) (by norm_num) (by norm_num) _).2 rfl

/-- A positive gap and an increasing competitor count require sharpening,
with no fitted coefficient. -/
theorem retention_gain_ge_one {original extended gap : ℝ} (ho : 0 < original)
    (he : original ≤ extended) (hd : 0 < gap) : 1 ≤ retentionGain original extended gap := by
  have hr : 1 ≤ extended / original := (le_div_iff₀ ho).2 (by simpa using he)
  have hl := Real.log_nonneg hr
  unfold retentionGain
  exact le_add_of_nonneg_right (div_nonneg hl hd.le)

example : 1 ≤ retentionGain 63 255 5 :=
  retention_gain_ge_one (by norm_num) (by norm_num) (by norm_num)

/-- The corresponding Q/K multiplier is the square root of the logit gain. -/
noncomputable def retentionLengthGain (original extended gap : ℝ) : ℝ :=
  Real.sqrt (retentionGain original extended gap)

/-- Scaling both Q and K by the derived factor gives the exact retained mass. -/
theorem retention_length_gain_preserves_mass {original extended gap : ℝ}
    (ho : 0 < original) (he : original ≤ extended) (hd : 0 < gap) :
    twoLevelMass extended gap (retentionLengthGain original extended gap ^ 2) =
      twoLevelMass original gap 1 := by
  have hg := retention_gain_ge_one ho he hd
  rw [retentionLengthGain, Real.sq_sqrt (by linarith : 0 ≤ retentionGain original extended gap)]
  exact (retention_gain_unique ho (lt_of_lt_of_le ho he) (ne_of_gt hd) _).2 rfl

example : twoLevelMass 255 5 (retentionLengthGain 63 255 5 ^ 2) = twoLevelMass 63 5 1 :=
  retention_length_gain_preserves_mass (by norm_num) (by norm_num) (by norm_num)

/-- Arbitrary logits need only a certified gap: the derived gain preserves
the original worst-case mass lower bound. It need not preserve the original
row's actual probability, which can exceed that lower bound. -/
theorem calibrated_mass_lower_bound {old n : ℕ} (hold : 0 < old) (hn : old ≤ n)
    (scores : Fin (n + 1) → ℝ) (target : Fin (n + 1)) {gap : ℝ} (hd : 0 < gap)
    (hgap : ∀ j, j ≠ target → scores j + gap ≤ scores target) :
    twoLevelMass old gap 1 ≤
      Precision.softmax (fun j => retentionGain old n gap * scores j) target := by
  have ho : (0 : ℝ) < old := Nat.cast_pos.mpr hold
  have he : (old : ℝ) ≤ n := Nat.cast_le.mpr hn
  have hg := retention_gain_ge_one ho he hd
  rw [← (retention_gain_unique ho (lt_of_lt_of_le ho he) (ne_of_gt hd)
    (retentionGain old n gap)).2 rfl]
  exact mass_lower_bound scores target gap _ (by linarith) hgap

example : twoLevelMass 1 1 1 ≤
    Precision.softmax (fun j : Fin 3 => retentionGain 1 2 1 * (if j = 0 then 1 else 0)) 0 :=
  by
    simpa only [Nat.cast_one, Nat.cast_ofNat] using
      calibrated_mass_lower_bound (old := 1) (n := 2) (by norm_num) (by norm_num)
        (fun j : Fin 3 => if j = 0 then 1 else 0) 0 (by norm_num : (0 : ℝ) < 1)
        (by intro j hj; fin_cases j <;> simp_all)

/-- Even with strictly positive target margins, one fixed gain cannot
preserve attention mass at both gaps one and two for the same length growth. -/
theorem no_common_retention_gain : ¬ ∃ gain : ℝ,
    twoLevelMass 2 1 gain = twoLevelMass 1 1 1 ∧
      twoLevelMass 2 2 gain = twoLevelMass 1 2 1 := by
  rintro ⟨gain, h1, h2⟩
  have he1 := (retention_gain_unique (by norm_num : (0 : ℝ) < 1)
    (by norm_num : (0 : ℝ) < 2) (by norm_num : (1 : ℝ) ≠ 0) gain).1 h1
  have he2 := (retention_gain_unique (by norm_num : (0 : ℝ) < 1)
    (by norm_num : (0 : ℝ) < 2) (by norm_num : (2 : ℝ) ≠ 0) gain).1 h2
  norm_num [retentionGain] at he1 he2
  have hl := Real.log_pos (by norm_num : (1 : ℝ) < 2)
  linarith

/-- Therefore no function of context lengths alone has the proposed
retention property for every positive-margin attention row. -/
theorem no_length_only_retention : ¬ ∃ gain : ℕ → ℕ → ℝ,
    ∀ original extended : ℕ, 0 < original → original ≤ extended →
      ∀ gap : ℝ, 0 < gap →
        twoLevelMass extended gap (gain (original + 1) (extended + 1)) =
          twoLevelMass original gap 1 := by
  rintro ⟨gain, h⟩
  apply no_common_retention_gain
  refine ⟨gain 2 3, ?_, ?_⟩
  · simpa using h 1 2 (by norm_num) (by norm_num) 1 (by norm_num)
  · simpa using h 1 2 (by norm_num) (by norm_num) 2 (by norm_num)

/-- RoPE rescaling may change the gap. Exact two-level calibration then
needs both the old and the new gap, not a silent equality assumption. -/
noncomputable def changedGapGain (original extended oldGap newGap : ℝ) : ℝ :=
  (oldGap + Real.log (extended / original)) / newGap

/-- The general exact calibration includes gap changes caused by the new
frequencies or activations, provided the new gap is nonzero. -/
theorem changed_gap_gain_unique {original extended newGap : ℝ}
    (ho : 0 < original) (he : 0 < extended) (hd : newGap ≠ 0) (oldGap gain : ℝ) :
    twoLevelMass extended newGap gain = twoLevelMass original oldGap 1 ↔
      gain = changedGapGain original extended oldGap newGap := by
  rw [mass_preservation_changed_gap_iff ho he]
  have hid : changedGapGain original extended oldGap newGap * newGap =
      oldGap + Real.log (extended / original) := by
    unfold changedGapGain
    exact div_mul_cancel₀ _ hd
  constructor
  · intro h
    exact mul_right_cancel₀ hd (h.trans hid.symm)
  · rintro rfl
    exact hid

example : twoLevelMass 255 3 (changedGapGain 63 255 5 3) = twoLevelMass 63 5 1 :=
  (changed_gap_gain_unique (by norm_num) (by norm_num) (by norm_num) _ _).2 rfl

end Transformer.RoPE.YaRN
