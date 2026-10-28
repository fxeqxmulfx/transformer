module

public import Transformer.Grokking.FlatSpectrum.BlockWitnessBest

/-!
# When the block witness beats `J`

Source: `review/flat_spectrum.md`, *Patterns*: "When `k + l > N`, the matrix
`X = J − t 1_R 1_Cᵀ` equals `J` on `Ω` and has rank 2. … The best `t` gives
`‖X‖²_S1 = N² − (k l − g)²/(k l) < N²`. At `k = 2`, `l = 60`, `N = 61` this is `60.912²`, the
norm ADMM found. For the cross, `k = l = N − 1`, it is item 9's `2√(N − 1)`."

At the best `t = 1 − g/(k l)`, `‖X‖²_S1 = m n − (k l − g)²/(k l)`, for `m = n = N` the source's
value (`nuclearNorm_onesSubBlock_best_sq`). Some real `t` has
`‖X‖_S1 < √(m n) = ‖J‖_S1` iff `m n < k n + l m` (`exists_nuclearNorm_onesSubBlock_lt_iff`),
for `N × N` iff `N < k + l` (`exists_nuclearNorm_onesSubBlock_lt_card_iff`): the witness of
Theorem E.3 (`exists_nuclearNorm_lt_of_block`) beats `J` exactly for the blocks it uses. At
`N = 61`, `k = 2`, `l = 60` the least norm is `√2 + √3540`, between `60.912` and `60.913`
(`isLeast_nuclearNorm_onesSubBlock_sixtyOne`). For `k = l = N − 1` and `N ≥ 2` it is
`2√(N − 1)` (`isLeast_nuclearNorm_onesSubBlock_cross`), item 9's least norm of the completions
from the cross (`isLeast_nuclearNorm_vecMulVec_cross`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- **The source's best value**, `review/flat_spectrum.md`, *Patterns*: under the hypotheses of
`isLeast_nuclearNorm_onesSubBlock`, `t = 1 − g/(k l)` gives
`‖J − t 1_R 1_Cᵀ‖²_S1 = m n − (k l − g)²/(k l)`, for `m = n = N` the source's
`N² − (k l − g)²/(k l)`. -/
theorem nuclearNorm_onesSubBlock_best_sq {R : Finset m} {C : Finset n} (hR : R.Nonempty)
    (hC : C.Nonempty)
    (h : (Fintype.card m * Fintype.card n : ℝ) ≤ #R * Fintype.card n + #C * Fintype.card m) :
    nuclearNorm (onesSubBlock R C
        (1 - √(#R * #C * (Fintype.card m - #R) * (Fintype.card n - #C)) / (#R * #C))) ^ 2 =
      Fintype.card m * Fintype.card n -
        (#R * #C - √(#R * #C * (Fintype.card m - #R) * (Fintype.card n - #C))) ^ 2 /
          (#R * #C) := by
  have hk : (0 : ℝ) < #R := by exact_mod_cast hR.card_pos
  have hl : (0 : ℝ) < #C := by exact_mod_cast hC.card_pos
  have hkM : (#R : ℝ) ≤ Fintype.card m := by exact_mod_cast card_le_univ R
  have hlN : (#C : ℝ) ≤ Fintype.card n := by exact_mod_cast card_le_univ C
  rw [nuclearNorm_onesSubBlock, radicand_best (mul_pos hk hl) (sqrt_le_mul_of_le hk.le hl.le h),
    Real.sq_sqrt (by rw [← sq_sqrt_add_sqrt hk hl hkM hlN]; positivity)]

example := nuclearNorm_onesSubBlock_best_sq (R := ({0, 1} : Finset (Fin 2)))
  (C := ({0} : Finset (Fin 2))) ⟨0, by decide⟩ ⟨0, by decide⟩ (by norm_num)

/-- **The block witness beats `J` exactly when the block is large**, `review/flat_spectrum.md`,
*Patterns*, generalized to `m × n`: some real `t` has
`‖J − t 1_R 1_Cᵀ‖_S1 < √(m n) = ‖J‖_S1` iff `m n < k n + l m`. -/
theorem exists_nuclearNorm_onesSubBlock_lt_iff (R : Finset m) (C : Finset n) :
    (∃ t, nuclearNorm (onesSubBlock R C t) < √(Fintype.card m * Fintype.card n)) ↔
      (Fintype.card m * Fintype.card n : ℝ) < #R * Fintype.card n + #C * Fintype.card m := by
  have hkM : (#R : ℝ) ≤ Fintype.card m := by exact_mod_cast card_le_univ R
  have hlN : (#C : ℝ) ≤ Fintype.card n := by exact_mod_cast card_le_univ C
  have hM : (0 : ℝ) ≤ Fintype.card m := Nat.cast_nonneg _
  have hN : (0 : ℝ) ≤ Fintype.card n := Nat.cast_nonneg _
  constructor
  · rintro ⟨t, ht⟩
    by_contra hle
    exact ((isLeast_nuclearNorm_onesSubBlock_of_le (not_lt.mp hle)).2 ⟨t, rfl⟩).not_gt ht
  intro h
  have hk : (0 : ℝ) < #R := by
    rcases (Nat.cast_nonneg (α := ℝ) #R).lt_or_eq with hk | hk
    · exact hk
    · rw [← hk] at h
      nlinarith [mul_le_mul_of_nonneg_right hlN hM]
  have hl : (0 : ℝ) < #C := by
    rcases (Nat.cast_nonneg (α := ℝ) #C).lt_or_eq with hl | hl
    · exact hl
    · rw [← hl] at h
      nlinarith [mul_le_mul_of_nonneg_right hkM hN]
  obtain ⟨⟨t, ht⟩, -⟩ := isLeast_nuclearNorm_onesSubBlock (card_pos.mp (by exact_mod_cast hk))
    (card_pos.mp (by exact_mod_cast hl)) h.le
  refine ⟨t, ?_⟩
  dsimp only at ht
  have hval := sq_sqrt_add_sqrt hk hl hkM hlN
  have hab : 0 ≤ √(#R * (Fintype.card n - #C)) + √(#C * (Fintype.card m - #R)) := by positivity
  rw [ht, ← Real.sqrt_sq hab, hval]
  refine Real.sqrt_lt_sqrt (by rw [← hval]; positivity) ?_
  have hg := sub_pos.mpr (sqrt_lt_mul_of_lt hk hl h)
  have hq : 0 < (#R * #C - √(#R * #C * (Fintype.card m - #R) * (Fintype.card n - #C))) ^ 2 /
      (#R * #C : ℝ) := by
    positivity
  linarith

/-- **Theorem E.3's witness, sharpened**, `review/flat_spectrum.md`, *Patterns*: for `N × N`,
some real `t` has `‖J − t 1_R 1_Cᵀ‖_S1 < N = ‖J‖_S1` iff `N < #R + #C`. -/
theorem exists_nuclearNorm_onesSubBlock_lt_card_iff (R C : Finset n) :
    (∃ t, nuclearNorm (onesSubBlock R C t) < Fintype.card n) ↔ Fintype.card n < #R + #C := by
  have h := exists_nuclearNorm_onesSubBlock_lt_iff R C
  rw [Real.sqrt_mul_self (Nat.cast_nonneg _)] at h
  rw [h]
  have hk : (#R : ℝ) ≤ Fintype.card n := by exact_mod_cast card_le_univ R
  have hl : (#C : ℝ) ≤ Fintype.card n := by exact_mod_cast card_le_univ C
  have hN : (0 : ℝ) ≤ Fintype.card n := Nat.cast_nonneg _
  constructor
  · intro h'
    by_contra hle
    have hle' : (#R + #C : ℝ) ≤ Fintype.card n := by exact_mod_cast not_lt.mp hle
    nlinarith [mul_le_mul_of_nonneg_left hle' hN]
  · intro h'
    have h'' : (Fintype.card n : ℝ) < #R + #C := by exact_mod_cast h'
    have hN' : 0 < (Fintype.card n : ℝ) := by linarith
    nlinarith [mul_lt_mul_of_pos_left h'' hN']

/-- **The failure of the survey**, `review/flat_spectrum.md`, *Patterns*: in `61 × 61`, two rows
`R` against `60` columns `C` give the least witness norm `√2 + √3540`, between `60.912` and
`60.913`. -/
theorem isLeast_nuclearNorm_onesSubBlock_sixtyOne {R C : Finset (Fin 61)} (hR : #R = 2)
    (hC : #C = 60) :
    IsLeast (Set.range fun t => nuclearNorm (onesSubBlock R C t)) (√2 + √3540) ∧
      60.912 < √2 + √3540 ∧ √2 + √3540 < 60.913 := by
  have h := isLeast_nuclearNorm_onesSubBlock (R := R) (C := C) (card_pos.mp (by omega))
    (card_pos.mp (by omega)) (by rw [hR, hC, Fintype.card_fin]; norm_num)
  have e : √(((2 : ℕ) : ℝ) * (((61 : ℕ) : ℝ) - ((60 : ℕ) : ℝ))) +
      √(((60 : ℕ) : ℝ) * (((61 : ℕ) : ℝ) - ((2 : ℕ) : ℝ))) = √2 + √3540 := by
    norm_num
  rw [hR, hC, Fintype.card_fin, e] at h
  refine ⟨h, ?_, ?_⟩
  · have h2 : (1.41421 : ℝ) < √2 := (Real.lt_sqrt (by norm_num)).mpr (by norm_num)
    have h3 : (59.49789 : ℝ) < √3540 := (Real.lt_sqrt (by norm_num)).mpr (by norm_num)
    linarith
  · have h2 : √2 < (1.41422 : ℝ) := (Real.sqrt_lt' (by norm_num)).mpr (by norm_num)
    have h3 : √3540 < (59.4979 : ℝ) := (Real.sqrt_lt' (by norm_num)).mpr (by norm_num)
    linarith

example := isLeast_nuclearNorm_onesSubBlock_sixtyOne (R := {0, 1}) (C := univ.erase 0)
  (by decide) (by simp)

/-- **The cross**, `review/flat_spectrum.md`, item 9: all rows but one against all columns but
one, `k = l = N − 1` with `N ≥ 2`, give the least witness norm `2√(N − 1)`, the least norm of
the completions from the cross (`isLeast_nuclearNorm_vecMulVec_cross`). -/
theorem isLeast_nuclearNorm_onesSubBlock_cross {R C : Finset n} (hR : #R + 1 = Fintype.card n)
    (hC : #C + 1 = Fintype.card n) (hN : 2 ≤ Fintype.card n) :
    IsLeast (Set.range fun t => nuclearNorm (onesSubBlock R C t))
      (2 * √(Fintype.card n - 1)) := by
  have hk : (#R : ℝ) = Fintype.card n - 1 := by
    rw [← hR]
    push_cast
    ring
  have hl : (#C : ℝ) = Fintype.card n - 1 := by
    rw [← hC]
    push_cast
    ring
  have hN' : (2 : ℝ) ≤ Fintype.card n := by exact_mod_cast hN
  have h := isLeast_nuclearNorm_onesSubBlock (R := R) (C := C) (card_pos.mp (by omega))
    (card_pos.mp (by omega)) (by rw [hk, hl]; nlinarith)
  rwa [hk, hl, sub_sub_cancel, mul_one, ← two_mul] at h

example := isLeast_nuclearNorm_onesSubBlock_cross (n := Fin 3) (R := univ.erase 0)
  (C := univ.erase 0) (by simp) (by simp) (by simp)

end Transformer.Grokking.FlatSpectrum
