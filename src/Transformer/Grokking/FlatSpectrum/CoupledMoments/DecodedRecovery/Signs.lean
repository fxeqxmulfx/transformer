module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.PairedLimits

/-!
# Signs behind a decoder that merges opposite latent states

Source: `review/flat_spectrum.md`, CM18. The decoders of this directory are fixed, known and
many-to-one: `d z = d w` exactly when `w = z` or `w = -z`, on the latent states `ZMod q` with
`q` odd. A decoded label therefore gives a latent state up to its sign, one unknown sign per
cell. The lemmas below are the whole arithmetic of the identification theorem:

* two rows that share two columns give the difference of their phases up to one common sign
  (`delta_sign`), when the two columns have different states and the four states of the
  rectangle do not sum to zero;
* that sign fixes the sign of every shared column (`column_sign`), when the rows differ;
* a further row is fixed by two of its cells in columns with different phases (`step_sign`).

The hypothesis `Odd q` enters through `x + x = 0 → x = 0`. On four latent states `step_sign`
fails (the last example).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery

open Schatten

variable {q : ℕ} {C : Type*}

/-- **The decoder class** (CM18): the labels of `z` and `w` agree exactly when `w = ± z`. -/
def PairedDecoder (d : ZMod q → C) : Prop := ∀ z w, d z = d w ↔ w = z ∨ w = -z

/-- A decoder of the class is even. -/
lemma PairedDecoder.even {d : ZMod q → C} (hd : PairedDecoder d) (z : ZMod q) : d (-z) = d z :=
  ((hd z (-z)).2 (Or.inr rfl)).symm

/-- A decoder of the class does not see a common sign. -/
lemma PairedDecoder.mul_sign {d : ZMod q → C} (hd : PairedDecoder d) {ε : ZMod q}
    (hε : ε = 1 ∨ ε = -1) (z : ZMod q) : d (ε * z) = d z := by
  rcases hε with rfl | rfl
  · rw [one_mul]
  · rw [neg_one_mul, hd.even]

/-- The folding decoder of CM16, on `2 k + 1` latent states, belongs to the class. -/
theorem pairedDecoder_fold (k : ℕ) : PairedDecoder (foldDecoder k) := by
  intro z w
  constructor
  · intro h
    obtain ⟨c, hc⟩ : ∃ c, foldDecoder k z = c := ⟨_, rfl⟩
    have hz := (foldDecoder_fiber k z c).1 hc
    have hw := (foldDecoder_fiber k w c).1 (h.symm.trans hc)
    rcases hz with hz | hz <;> rcases hw with hw | hw <;> rw [hz, hw]
    · exact Or.inl rfl
    · exact Or.inr rfl
    · exact Or.inr (neg_neg _).symm
    · exact Or.inl rfl
  · rintro (rfl | rfl)
    · rfl
    · exact (foldDecoder_even k z).symm

/-- The two-class decoder of CM16 on three latent states belongs to the class. -/
theorem pairedDecoder_bool : PairedDecoder pairedBoolDecoder := by
  unfold PairedDecoder
  decide

/-- One column of two rows leaves four candidates for the difference `δ'` of the rows of a
second fitting table: the true `± δ` and the false `± (2 u + δ)`. -/
lemma delta_cases {u δ u' δ' : ZMod q} (h : u' = u ∨ u' = -u)
    (hδ : u' + δ' = u + δ ∨ u' + δ' = -(u + δ)) :
    δ' = δ ∨ δ' = -δ ∨ δ' = u + u + δ ∨ δ' = -(u + u + δ) := by
  rcases h with rfl | rfl <;> rcases hδ with h | h
  · exact Or.inl (by linear_combination h)
  · exact Or.inr (Or.inr (Or.inr (by linear_combination h)))
  · exact Or.inr (Or.inr (Or.inl (by linear_combination h)))
  · exact Or.inr (Or.inl (by linear_combination h))

/-- **Two columns fix the difference of two rows up to one sign** (CM18): the false candidates
of the two columns are disjoint when `u₁ ≠ u₂` and `u₁ + u₂ + δ ≠ 0`. -/
lemma delta_sign (hq : Odd q) {u₁ u₂ δ u₁' u₂' δ' : ZMod q}
    (h₁ : u₁' = u₁ ∨ u₁' = -u₁) (h₁δ : u₁' + δ' = u₁ + δ ∨ u₁' + δ' = -(u₁ + δ))
    (h₂ : u₂' = u₂ ∨ u₂' = -u₂) (h₂δ : u₂' + δ' = u₂ + δ ∨ u₂' + δ' = -(u₂ + δ))
    (hne : u₁ ≠ u₂) (hsw : u₁ + u₂ + δ ≠ 0) : δ' = δ ∨ δ' = -δ := by
  have h2 (x : ZMod q) (hx : x + x = 0) : x = 0 := (ZMod.add_self_eq_zero_iff_eq_zero hq).1 hx
  rcases delta_cases h₁ h₁δ with h | h | h | h
  · exact Or.inl h
  · exact Or.inr h
  · rcases delta_cases h₂ h₂δ with k | k | k | k
    · exact Or.inl k
    · exact Or.inr k
    · exact absurd (sub_eq_zero.1 (h2 (u₁ - u₂) (by linear_combination k - h))) hne
    · exact absurd (h2 (u₁ + u₂ + δ) (by linear_combination k - h)) hsw
  · rcases delta_cases h₂ h₂δ with k | k | k | k
    · exact Or.inl k
    · exact Or.inr k
    · exact absurd (h2 (u₁ + u₂ + δ) (by linear_combination h - k)) hsw
    · exact absurd (sub_eq_zero.1 (h2 (u₁ - u₂) (by linear_combination h - k))) hne

/-- **The sign of the rows is the sign of every shared column** (CM18), when the rows have
different phases. -/
lemma column_sign (hq : Odd q) {u δ u' ε : ZMod q} (hε : ε = 1 ∨ ε = -1) (hδ : δ ≠ 0)
    (h : u' = u ∨ u' = -u) (hδ' : u' + ε * δ = u + δ ∨ u' + ε * δ = -(u + δ)) :
    u' = ε * u := by
  have h2 (x : ZMod q) (hx : x + x = 0) : x = 0 := (ZMod.add_self_eq_zero_iff_eq_zero hq).1 hx
  rcases hε with rfl | rfl <;> rcases h with h | h
  · linear_combination h
  · rcases hδ' with k | k
    · linear_combination h - 2 * h2 u (by linear_combination h - k)
    · exact absurd (h2 δ (by linear_combination k - h)) hδ
  · rcases hδ' with k | k
    · exact absurd (h2 δ (by linear_combination h - k)) hδ
    · linear_combination h + 2 * h2 u (by linear_combination k - h)
  · linear_combination h

/-- **Two cells in columns with different phases fix a row** (CM18): a state `x'` that has the
labels of `x` beside both `a` and `b` is `x`. The reflected candidates `-x - 2a` and `-x - 2b`
differ. -/
lemma step_sign (hq : Odd q) {x a b x' : ZMod q} (hab : a ≠ b)
    (h₁ : x' + a = x + a ∨ x' + a = -(x + a)) (h₂ : x' + b = x + b ∨ x' + b = -(x + b)) :
    x' = x := by
  rcases h₁ with h | h
  · exact add_right_cancel h
  · rcases h₂ with k | k
    · exact add_right_cancel k
    · exact absurd (sub_eq_zero.1 ((ZMod.add_self_eq_zero_iff_eq_zero hq).1
        (show a - b + (a - b) = 0 by linear_combination h - k))) hab

/-- The same in labels: the phase of a row is the only state with its two labels. -/
theorem step_unique (hq : Odd q) {d : ZMod q → C} (hd : PairedDecoder d) {x a b x' : ZMod q}
    (hab : a ≠ b) (h₁ : d (x' + a) = d (x + a)) (h₂ : d (x' + b) = d (x + b)) : x' = x :=
  step_sign hq hab ((hd _ _).1 h₁.symm) ((hd _ _).1 h₂.symm)

/-! The hypotheses are met on five latent states with the folding decoder. -/

example := (pairedDecoder_fold 2).even 1
example := (pairedDecoder_fold 2).mul_sign (Or.inr rfl) 1
example := delta_cases (u := (1 : ZMod 5)) (δ := 1) (u' := 4) (δ' := 4) (by decide) (by decide)
example := delta_sign (q := 5) (by decide) (u₁ := 0) (u₂ := 3) (δ := 1) (u₁' := 0) (u₂' := 2)
  (δ' := 4) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
example := column_sign (q := 5) (by decide) (u := 3) (δ := 1) (u' := 2) (ε := -1)
  (Or.inr rfl) (by decide) (by decide) (by decide)
example := step_sign (q := 5) (by decide) (x := 2) (a := 0) (b := 3) (x' := 2) (by decide)
  (by decide) (by decide)
example := step_unique (q := 5) (by decide) (pairedDecoder_fold 2) (x := 2) (a := 0) (b := 3)
  (x' := 2) (by decide) rfl rfl

/-- On four latent states two cells do not fix a row: `Odd q` cannot be dropped. -/
example : ∃ x a b x' : ZMod 4, a ≠ b ∧ (x' + a = x + a ∨ x' + a = -(x + a)) ∧
    (x' + b = x + b ∨ x' + b = -(x + b)) ∧ x' ≠ x :=
  ⟨1, 0, 2, 3, by decide, by decide, by decide, by decide⟩

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery
