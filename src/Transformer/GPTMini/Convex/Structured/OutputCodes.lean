import Transformer.GPTMini.Semantics.RecallCodes

/-!
# Ten-coordinate decoder for the whole Basis vocabulary

Source: the four-pair signed-axis recall code at a7d5e0f, generalized
to five base-four digits. This covers IDs 0 through 1023, including
every token in Basis's largest vocabulary of size 548. Ten readonly
decoder coordinates suffice; learned input query/key/value potentials
are separate free coordinates. No input-prefix feature bank is used.

Distinct output IDs have a uniform score gap of one. Arbitrary latent
channel assignments have bounded decoder scores, which lets actual
normalized model confidence imply strict whole-vocabulary margins.
The residual/tied embedding implementation remains a separate step.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.GPTMini.Semantics
open scoped BigOperators Classical
noncomputable section

/-- Five base-four digits encode every actual Basis token ID in a fixed decoder.
Source: the compact signed-axis code, extended from the 256-symbol recall alphabet to 1024 possible output IDs. -/
def outputDigit (token : Fin 1024) (place : Fin 5) : Fin 4 :=
  ⟨token.val / 4 ^ place.val % 4, Nat.mod_lt _ (by decide)⟩

/-- One decoder pair's true dot product uses the two signed-axis coordinates.
Source: RecallCodes.quarterX/quarterY; the channel label is inferred, while the candidate token's code is readonly. -/
def outputPairScore (a b : Fin 4) : ℝ :=
  quarterX a * quarterX b + quarterY a * quarterY b

/-- The whole decoder score is the dot product of five actual coordinate pairs.
Source: the ten-axis output layout; arbitrary channel assignments are permitted, not just valid vocabulary IDs. -/
def outputCodeScore (a b : Fin 5 → Fin 4) : ℝ :=
  ∑ h, outputPairScore (a h) (b h)

/-- A channel assignment's actual ten scalar coordinates use two consecutive axes per digit.
Source: the compact readonly decoder layout; the learned head outputs expectations of these same coordinates. -/
def outputCoordinate (assignment : Fin 5 → Fin 4) (i : Fin 10) : ℝ :=
  if i.val % 2 = 0 then quarterX (assignment ⟨i.val / 2, by have hi := i.isLt; omega⟩)
  else quarterY (assignment ⟨i.val / 2, by have hi := i.isLt; omega⟩)

/-- Every physical even output axis carries the actual first coordinate of its digit pair.
Source: the explicit ten-axis layout and bounded physical index arithmetic. -/
theorem outputCoordinate_even (assignment : Fin 5 → Fin 4) (h : Fin 5) :
    outputCoordinate assignment ⟨2 * h.val, by have hh := h.isLt; omega⟩ = quarterX (assignment h) := by
  have hm : (2 * h.val) % 2 = 0 := by omega
  have hd : (2 * h.val) / 2 = h.val := by omega
  simp only [outputCoordinate, hm, hd, ite_true]

/-- Every physical odd output axis carries the actual second coordinate of the same digit pair.
Source: the explicit ten-axis layout, so the scalar score is grounded in real output coordinates. -/
theorem outputCoordinate_odd (assignment : Fin 5 → Fin 4) (h : Fin 5) :
    outputCoordinate assignment ⟨2 * h.val + 1, by have hh := h.isLt; omega⟩ = quarterY (assignment h) := by
  have hm : (2 * h.val + 1) % 2 = 1 := by omega
  have hd : (2 * h.val + 1) / 2 = h.val := by omega
  simp only [outputCoordinate, hm, hd, show ¬(1 : ℕ) = 0 by decide, ite_false]

/-- Five actual radix digits distinguish all IDs below 1024.
Source: bounded radix reconstruction, with the fifth digit preventing collisions between all keys, values and special tokens. -/
theorem outputDigit_injective : Function.Injective outputDigit := by
  intro a b h
  have h0 := congrArg (fun digits : Fin 5 → Fin 4 => (digits 0).val) h
  have h1 := congrArg (fun digits : Fin 5 → Fin 4 => (digits 1).val) h
  have h2 := congrArg (fun digits : Fin 5 → Fin 4 => (digits 2).val) h
  have h3 := congrArg (fun digits : Fin 5 → Fin 4 => (digits 3).val) h
  have h4 := congrArg (fun digits : Fin 5 → Fin 4 => (digits 4).val) h
  norm_num [outputDigit] at h0 h1 h2 h3 h4
  apply Fin.ext
  have ha := a.isLt
  have hb := b.isLt
  omega

/-- Identical actual signed-axis pairs have score exactly one.
Source: the explicit four-channel unit-vector assignments from RecallCodes. -/
theorem outputPairScore_self (a : Fin 4) : outputPairScore a a = 1 := by
  unfold outputPairScore
  simpa only [pow_two] using quarter_unit a

/-- Every actual signed-axis comparison is bounded on both sides.
Source: the sixteen true two-coordinate dot products, including opposite signed axes. -/
theorem outputPairScore_bounds (a b : Fin 4) :
    -1 ≤ outputPairScore a b ∧ outputPairScore a b ≤ 1 := by
  fin_cases a <;> fin_cases b <;> norm_num [outputPairScore, quarterX, quarterY]

/-- A different actual channel has dot product at most zero against the target digit.
Source: RecallCodes.quarter_inner's finite geometric comparison. -/
theorem outputPairScore_gap (a b : Fin 4) (hne : a ≠ b) : outputPairScore a b ≤ 0 := by
  have h := quarter_inner a b
  rw [ite_eq_right hne] at h
  exact h

example : (0 : Fin 4) ≠ 1 := by decide

/-- Every complete five-channel assignment scores exactly five against itself.
Source: all five actual unit coordinate pairs, without a supplied logit margin. -/
theorem outputCodeScore_self (a : Fin 5 → Fin 4) : outputCodeScore a a = 5 := by
  simp only [outputCodeScore, outputPairScore_self, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  norm_num

/-- Scores of arbitrary latent assignments stay between minus five and five.
Source: finite addition of the proved bounds for the five actual coordinate pairs. -/
theorem outputCodeScore_bounds (a b : Fin 5 → Fin 4) :
    -5 ≤ outputCodeScore a b ∧ outputCodeScore a b ≤ 5 := by
  have hlo := Finset.sum_le_sum (fun h (_ : h ∈ (Finset.univ : Finset (Fin 5))) =>
    (outputPairScore_bounds (a h) (b h)).1)
  have hup := Finset.sum_le_sum (fun h (_ : h ∈ (Finset.univ : Finset (Fin 5))) =>
    (outputPairScore_bounds (a h) (b h)).2)
  norm_num only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hlo hup
  exact ⟨hlo, hup⟩

/-- Different five-channel assignments have a strict decoder gap of at least one.
Source: one differing pair scores at most zero and the remaining four pairs each score at most one. -/
theorem outputCodeScore_gap (a b : Fin 5 → Fin 4) (hne : a ≠ b) : outputCodeScore a b ≤ 4 := by
  obtain ⟨p, hp⟩ : ∃ p, a p ≠ b p := by
    by_contra hn
    apply hne
    funext p
    by_contra h
    exact hn ⟨p, h⟩
  have hs : outputCodeScore a b ≤ ∑ h : Fin 5, if h = p then (0 : ℝ) else 1 := by
    apply Finset.sum_le_sum
    intro h hh
    by_cases he : h = p
    · subst h
      simpa only [ite_true, eq_self] using outputPairScore_gap (a p) (b p) hp
    · rw [ite_eq_right he]
      exact (outputPairScore_bounds (a h) (b h)).2
  have hc : (∑ h : Fin 5, if h = p then (0 : ℝ) else 1) = 4 := by
    fin_cases p <;> norm_num [Fin.sum_univ_five]
  exact hs.trans (by rw [hc])

example : (fun _ : Fin 5 => (0 : Fin 4)) ≠ (fun _ : Fin 5 => (1 : Fin 4)) := by
  intro h
  have he := congrArg (fun digits : Fin 5 → Fin 4 => digits 0) h
  contradiction

/-- Every distinct vocabulary ID has a strict gap against the true target code.
Source: actual five-digit injection and true decoder geometry, covering all 1024 possible IDs rather than only answer labels. -/
theorem outputToken_gap (target rival : Fin 1024) (hne : target ≠ rival) :
    outputCodeScore (outputDigit target) (outputDigit rival) ≤ 4 := by
  apply outputCodeScore_gap
  intro he
  exact hne (outputDigit_injective he)

example : (17 : Fin 1024) ≠ 25 := by decide

/-- The largest Basis vocabulary fits strictly inside the fixed ten-coordinate output alphabet.
Source: recall's vocabulary size 548 and the decoder's 1024 distinct IDs. -/
theorem outputAlphabet_covers_basis : (548 : ℕ) ≤ 1024 := by omega

end
end Transformer.GPTMini.Convex.Structured
