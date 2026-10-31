module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.PoweredDivision

/-!
# Feature rank, score rank, and indicator rank are different

Source: `review/flat_spectrum.md`, *Joint feature recovery and classification*, item 7.
A bank of r separable complex features gives a real score matrix of rank at most 2r for
each class. This does not bound the rank of the discrete indicator after argmax: division's
identity-class indicator has full rank on every finite group. For every prime, a faithful
character classifies division with real score matrices of rank at most two, while that
indicator has rank p-1. This concerns an explicit score representation, not the numerical
rank of a trained network's logits or its required width.

The factorization keeps the operand indices visible. For class c, set
`B_r(j) = v_r(j) C(c,r)`. Its contribution to the score is
`w_r Re(u_r(i)) Re(B_r(j)) + w_r Im(u_r(i)) Im(B_r(j))`.
Concatenating the real and imaginary factors gives a matrix product with 2r columns.
The class-dependent right factor changes with c, but the left factor is shared.

The indicator has different entries: it is one exactly where division returns the
identity and zero elsewhere. Its diagonal entries are one, and every off-diagonal
entry is zero. Computing this indicator from correct argmax decisions does not require
the real-valued score matrix to equal it.

The final existence theorem quantifies over every natural exponent with one character.
Its indicator-rank conjunct refers specifically to ordinary division, independently
of that exponent. No formula for the indicator rank of higher powers is asserted here.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Matrix
open scoped BigOperators

variable {m n R A : Type*} [Fintype n] [Fintype R]

/-- Shared real row factors for a bank of complex features, source item 7. -/
def realLeft (w : R → ℝ) (u : R → m → ℂ) : Matrix m (R ⊕ R) ℝ :=
  fun i => Sum.elim (fun r => w r * (u r i).re) (fun r => w r * (u r i).im)

/-- Class-dependent real column factors, source item 7. -/
def realRight (v : R → n → ℂ) (C : A → R → ℂ) (c : A) : Matrix n (R ⊕ R) ℝ :=
  fun j => Sum.elim (fun r => (v r j * C c r).re) (fun r => (v r j * C c r).im)

omit [Fintype n] in
/-- **Real factorization of the score matrix**, source item 7. The two real coordinates
of each complex feature suffice, with shared row factors for every class. -/
theorem score_matrix_factorization (w : R → ℝ) (C : A → R → ℂ)
    (u : R → m → ℂ) (v : R → n → ℂ) (c : A) :
    (fun i j => score w C (fun r => u r i * star (v r j)) c) =
      realLeft w u * (realRight v C c)ᵀ := by
  ext i j
  simp only [score, Matrix.mul_apply, Matrix.transpose_apply, Fintype.sum_sum_type,
    realLeft, realRight, Sum.elim_inl, Sum.elim_inr]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro r _
  simp only [Complex.star_def, Complex.mul_re, Complex.mul_im, Complex.conj_re, Complex.conj_im]
  ring

/-- A class score from r separable complex features has real rank at most 2r, source
item 7. This bound needs no unit norms, positive weights, or separation assumption. -/
theorem rank_score_matrix_le (w : R → ℝ) (C : A → R → ℂ)
    (u : R → m → ℂ) (v : R → n → ℂ) (c : A) :
    Matrix.rank (fun i j => score w C (fun r => u r i * star (v r j)) c) ≤
      2 * Fintype.card R := by
  rw [score_matrix_factorization]
  apply (rank_mul_le_left _ _).trans
  simpa only [Fintype.card_sum, two_mul] using rank_le_card_width (realLeft w u)

/-- The identity-class indicator of division, source item 7. It is a full output table,
not a scalar character readout or a matrix of logits. -/
def divisionIndicator (G : Type*) [Group G] [DecidableEq G] : Matrix G G ℝ :=
  fun a b => if a / b = 1 then 1 else 0

/-- Division's identity-class indicator has **full rank** on any finite group, source
item 7. Every row has its sole one in the matching column, so the matrix is the identity. -/
theorem rank_divisionIndicator (G : Type*) [Group G] [Fintype G] [DecidableEq G] :
    (divisionIndicator G).rank = Fintype.card G := by
  have h : divisionIndicator G = 1 := by
    ext a b
    simp only [divisionIndicator, div_eq_one, Matrix.one_apply]
  rw [h, Matrix.rank_one]

/-- The explicit powered-division scores use at most two real rank-one terms per
character, source item 7. The exponent changes the factors, not this rank bound. -/
theorem rank_power_scores_le {p : ℕ} [Fact p.Prime] (w : R → ℝ)
    (χ : R → MulChar (ZMod p) ℂ) (k : ℕ) (c : (ZMod p)ˣ) :
    Matrix.rank (fun a b => score w (unitCode χ) (fun r => powerKernel (χ r) k a b) c) ≤
      2 * Fintype.card R := by
  have h (a b : (ZMod p)ˣ) : (fun r => powerKernel (χ r) k a b) =
      fun r => (χ r a ^ k) * star (χ r b ^ k) := by
    funext r
    rw [powerKernel_eq_outer, vecMulVec_apply, star_pow]
  simp_rw [h]
  exact rank_score_matrix_le (m := (ZMod p)ˣ) (n := (ZMod p)ˣ) w (unitCode χ)
    (fun r a => χ r a ^ k) (fun r b => χ r b ^ k) c

/-- **Arbitrarily many answer classes through low-rank scores**, source item 7. For every
prime, one complex character gives real score rank at most two and strict correct decoding
at every exponent. At exponent one, the identity-class indicator still has rank p-1. -/
theorem exists_low_rank_power_classifier {p : ℕ} [Fact p.Prime] :
    ∃ χ : MulChar (ZMod p) ℂ,
      (∀ (k : ℕ) (c : (ZMod p)ˣ),
        Matrix.rank (fun a b => score (fun _ : Unit => 1) (unitCode (fun _ => χ))
          (fun _ => powerKernel χ k a b) c) ≤ 2) ∧
      (∀ (k : ℕ) (a b c : (ZMod p)ˣ), c ≠ (a / b) ^ k →
        score (fun _ : Unit => 1) (unitCode (fun _ => χ)) (fun _ => powerKernel χ k a b) c <
          score (fun _ : Unit => 1) (unitCode (fun _ => χ))
            (fun _ => powerKernel χ k a b) ((a / b) ^ k)) ∧
      (divisionIndicator (ZMod p)ˣ).rank = p - 1 := by
  obtain ⟨χ, hχ⟩ := exists_injective_mulChar (p := p)
  refine ⟨χ, ?_, ?_, ?_⟩
  · intro k c
    simpa using rank_power_scores_le (fun _ : Unit => (1 : ℝ)) (fun _ => χ) k c
  · intro k a b c hc
    simp_rw [powerKernel_eq_outer, vecMulVec_apply]
    exact power_scores_correct (fun _ : Unit => zero_le_one) (fun _ => χ)
      (fun a b hab => ⟨(), zero_lt_one, fun h => hab (Units.ext (hχ h))⟩) k a b c hc
  · rw [rank_divisionIndicator, Fintype.card_units, ZMod.card]

example := score_matrix_factorization (m := Bool) (n := Bool)
  (fun _ : Bool => (1 / 2 : ℝ)) binaryCode
  (fun _ a => if a then (1 : ℂ) else -1) (fun _ b => if b then (1 : ℂ) else -1) (true, false)

example := rank_score_matrix_le (m := Bool) (n := Bool)
  (fun _ : Bool => (1 / 2 : ℝ)) binaryCode
  (fun _ a => if a then (1 : ℂ) else -1) (fun _ b => if b then (1 : ℂ) else -1) (true, false)

example : (divisionIndicator (ZMod 3)ˣ).rank = 2 := by
  rw [rank_divisionIndicator, Fintype.card_units, ZMod.card]

example := rank_power_scores_le (p := 3) (fun _ : Bool => (1 / 2 : ℝ)) (fun _ => 1) 0 1
example := exists_low_rank_power_classifier (p := 3)

section

set_option maxRecDepth 4096 in
local instance : Fact (Nat.Prime 193) := ⟨by decide⟩

example := exists_low_rank_power_classifier (p := 193)
example : (divisionIndicator (ZMod 193)ˣ).rank = 192 := by
  rw [rank_divisionIndicator, Fintype.card_units, ZMod.card]

end

end Transformer.Grokking.FlatSpectrum.FeatureCodes
