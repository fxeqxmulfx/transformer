module

public import Transformer.Grokking.FlatSpectrum.FlatCapacity
public import Transformer.Grokking.FlatSpectrum.Roots
public import Mathlib.Data.Matrix.ColumnRowPartitioned

/-!
# Balanced factors: cost, rank and a flat-target error bound

Source: `review/flat_spectrum.md`, *Balancedness as a capacity certificate*, B1–B2.
This generalizes the identities in WD14.6: the hypothesis is the Gram equality
`Uᴴ U = Vᴴ V` itself, without assuming stationarity or a particular loss.

Balanced factors have equal Frobenius norms, their average squared norm is the
nuclear norm of their product, and both factors and their stack have the rank
of the product. R3 therefore bounds the error against a flat target by the
actual factor rank, even when the allocated width is larger. These identities
do not assert optimality, learning unseen entries, or convergence of minibatch
updates. B3 gives an explicit obstruction to the first conclusion.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k]
  [DecidableEq m] [DecidableEq n] {U : Matrix m k ℂ} {V : Matrix n k ℂ}

omit [DecidableEq m] [DecidableEq n] in
/-- **Equal factor norms from balance**, B1; WD14.6 without stationarity. -/
theorem frobeniusSq_eq_of_balanced (hbal : Uᴴ * U = Vᴴ * V) :
    frobeniusSq U = frobeniusSq V := by
  rw [frobeniusSq_eq_re_trace, frobeniusSq_eq_re_trace, hbal]

omit [DecidableEq m] [DecidableEq n] in
/-- **The Gram matrix of a balanced product**, B1: its positive root is `V Vᴴ`. -/
theorem gram_mul_conjTranspose_of_balanced (hbal : Uᴴ * U = Vᴴ * V) :
    (U * Vᴴ)ᴴ * (U * Vᴴ) = (V * Vᴴ)ᴴ * (V * Vᴴ) := by
  simp only [conjTranspose_mul, conjTranspose_conjTranspose, Matrix.mul_assoc]
  rw [← Matrix.mul_assoc Uᴴ, hbal, Matrix.mul_assoc]

omit [DecidableEq m] in
/-- **Balanced weight-decay cost is the nuclear norm**, B1; the equality in Lemma 1
needs only balance, without any claim that the product minimizes a loss. -/
theorem nuclearNorm_mul_conjTranspose_of_balanced (hbal : Uᴴ * U = Vᴴ * V) :
    nuclearNorm (U * Vᴴ) = (frobeniusSq U + frobeniusSq V) / 2 := by
  have hPX : (V * Vᴴ) * (V * Vᴴ) = (U * Vᴴ)ᴴ * (U * Vᴴ) := by
    rw [gram_mul_conjTranspose_of_balanced hbal, conjTranspose_mul,
      conjTranspose_conjTranspose]
  have htrace := nuclearNorm_eq_trace_of_mul_self (posSemidef_self_mul_conjTranspose V) hPX
  have hnorm : (V * Vᴴ).trace.re = frobeniusSq V := by
    rw [Matrix.trace_mul_comm, frobeniusSq_eq_re_trace]
  have hcost : nuclearNorm (U * Vᴴ) = frobeniusSq V := by
    rw [← hnorm, ← htrace, Complex.ofReal_re]
  rw [hcost, frobeniusSq_eq_of_balanced hbal]
  ring

omit [DecidableEq m] [DecidableEq n] in
/-- **Balanced factors have the rank of their product**, B1. -/
theorem rank_mul_conjTranspose_of_balanced (hbal : Uᴴ * U = Vᴴ * V) :
    (U * Vᴴ).rank = U.rank ∧ (U * Vᴴ).rank = V.rank := by
  have hUV : U.rank = V.rank := by
    rw [← rank_conjTranspose_mul_self U, hbal, rank_conjTranspose_mul_self]
  have hX : (U * Vᴴ).rank = V.rank := by
    rw [← rank_conjTranspose_mul_self (U * Vᴴ), gram_mul_conjTranspose_of_balanced hbal,
      rank_conjTranspose_mul_self, rank_self_mul_conjTranspose]
  exact ⟨hX.trans hUV.symm, hX⟩

omit [DecidableEq m] [DecidableEq n] in
/-- **The stack has no additional rank at balance**, B1: its Gram matrix is twice
the Gram matrix of either factor. This does not say gradient steps preserve balance. -/
theorem rank_fromRows_of_balanced (hbal : Uᴴ * U = Vᴴ * V) :
    (fromRows U V).rank = U.rank := by
  rw [← rank_conjTranspose_mul_self (fromRows U V),
    conjTranspose_fromRows_eq_fromCols_conjTranspose, fromCols_mul_fromRows, ← hbal]
  have htwice : Uᴴ * U + Uᴴ * U = (2 : ℂ) • (Uᴴ * U) := by
    simp [two_smul]
  rw [htwice, rank_smul_of_mem_nonZeroDivisors _
    (mem_nonZeroDivisors_of_ne_zero (by norm_num : (2 : ℂ) ≠ 0)),
    rank_conjTranspose_mul_self]

omit [DecidableEq m] [DecidableEq n] in
/-- **A balanced stack cannot have full row rank when the second factor has a
row**, B1. Thus exact balanced initialization does not meet WD5's full-stack
rank hypothesis. This does not preclude successful training by another result. -/
theorem rank_fromRows_lt_card_of_balanced [Nonempty n] (hbal : Uᴴ * U = Vᴴ * V) :
    (fromRows U V).rank < Fintype.card (m ⊕ n) := by
  rw [rank_fromRows_of_balanced hbal, Fintype.card_sum]
  have hm := rank_le_card_height U
  have hn : 0 < Fintype.card n := Fintype.card_pos
  omega

/-- **Flat-spectrum error uses the actual balanced factor rank**, B2. The target
has squared singular value `c > 0`; the conclusion is an error lower bound, not
a statement that missing labels are identified or that training reaches it. -/
theorem IsFlat.mul_sub_factor_rank_le_frobeniusSq_sub {K : Matrix m n ℂ} {c : ℝ}
    (hc : 0 < c) (hflat : IsFlat K c) (hbal : Uᴴ * U = Vᴴ * V) :
    c * ((min (Fintype.card m) (Fintype.card n) : ℕ) - U.rank) ≤
      frobeniusSq (K - U * Vᴴ) := by
  have herror := hflat.mul_sub_rank_le_frobeniusSq_sub hc (U * Vᴴ)
  rw [(rank_mul_conjTranspose_of_balanced hbal).1] at herror
  exact herror

/-- **An error budget requires actual factor rank**, B2. This is a necessary
capacity condition, with no sufficiency or generalization conclusion. -/
theorem IsFlat.le_factor_rank_of_error_le {K : Matrix m n ℂ} {c ε : ℝ}
    (hc : 0 < c) (hflat : IsFlat K c) (hbal : Uᴴ * U = Vᴴ * V)
    (herr : frobeniusSq (K - U * Vᴴ) ≤ ε) :
    (min (Fintype.card m) (Fintype.card n) : ℕ) - ε / c ≤ (U.rank : ℝ) := by
  have h := (hflat.mul_sub_factor_rank_le_frobeniusSq_sub hc hbal).trans herr
  have hr : ((min (Fintype.card m) (Fintype.card n) : ℕ) : ℝ) - U.rank ≤ ε / c := by
    apply (le_div_iff₀ hc).2
    nlinarith
  linarith

/-! Nonzero witnesses for every balance hypothesis, and a flat target with zero
balanced factors. The latter shows the error bound is attained. -/

example : frobeniusSq (1 : Matrix (Fin 2) (Fin 2) ℂ) =
    frobeniusSq (1 : Matrix (Fin 2) (Fin 2) ℂ) :=
  frobeniusSq_eq_of_balanced rfl

example := gram_mul_conjTranspose_of_balanced
  (U := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (V := 1) rfl

example := nuclearNorm_mul_conjTranspose_of_balanced
  (U := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (V := 1) rfl

example := rank_mul_conjTranspose_of_balanced
  (U := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (V := 1) rfl

example := rank_fromRows_of_balanced
  (U := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (V := 1) rfl

example := rank_fromRows_lt_card_of_balanced
  (U := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (V := 1) rfl

example : (2 : ℝ) ≤ frobeniusSq (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
  have h := isFlat_one.mul_sub_factor_rank_le_frobeniusSq_sub one_pos
    (U := (0 : Matrix (Fin 2) (Fin 2) ℂ)) (V := 0) rfl
  simpa using h

example := isFlat_one.le_factor_rank_of_error_le (n := Fin 2) one_pos
  (U := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (V := 1) rfl (ε := 0)
  (by simp [frobeniusSq])

end Transformer.Grokking.FlatSpectrum
