/-
# Phase model — orthogonality of the characters

The phases `2πjx/n` of a character of `ℤ/n` are additive up to `2πℤ`
(`exists_characterPhase_sub`), and the characters are orthogonal:
`Σ_{j<n} e^{2πijd/n}` is `n` for `d = 0` and `0` otherwise
(`sum_exp_characterPhase`), so `Σ_j e^{i(2πja/n - 2πjb/n)}` is `n` for
`a = b` and `0` otherwise (`sum_exp_characterPhase_sub`), and
`Σ_j e^{i(2πjb/n + 2πjc/n - 2πja/n)}` is `n` for `a = b + c` and `0`
otherwise (`sum_exp_characterPhase_add_sub`).
-/

import Transformer.PhaseModel.Division
import Mathlib.Analysis.SpecialFunctions.Complex.Log
import Mathlib.Algebra.Field.GeomSum

open Real ComplexConjugate

namespace Transformer.PhaseModel

variable {n : ℕ}

/-- **The phases of a character are additive up to `2πℤ`.** -/
theorem exists_characterPhase_sub (j : ℤ) (a b : Fin n) : ∃ k : ℤ,
    (characterPhase n j).ofLp a - (characterPhase n j).ofLp b =
      (characterPhase n j).ofLp (a - b) + k * (2 * π) := by
  obtain ⟨k, hk⟩ := exists_inner_intVec_divisionRow_characterPhase j (a, b)
  rw [inner_intVec_divisionRow] at hk
  exact ⟨k, by dsimp only at hk; linarith⟩

/-- `e^{i(x + 2πk)} = e^{ix}`. -/
theorem exp_ofReal_add_int_mul_two_pi_mul_I (x : ℝ) (k : ℤ) :
    Complex.exp (↑(x + k * (2 * π)) * Complex.I) = Complex.exp (↑x * Complex.I) := by
  rw [show (↑(x + k * (2 * π)) : ℂ) * Complex.I = ↑x * Complex.I + k * (2 * π * Complex.I) by
    push_cast; ring, Complex.exp_add, Complex.exp_int_mul_two_pi_mul_I, mul_one]

/-- `conj e^{iθ} = e^{-iθ}`. -/
theorem conj_exp_ofReal_mul_I (θ : ℝ) :
    conj (Complex.exp (↑θ * Complex.I)) = Complex.exp (↑(-θ) * Complex.I) := by
  rw [← Complex.exp_conj, map_mul, Complex.conj_ofReal, Complex.conj_I]
  push_cast
  ring_nf

variable [NeZero n]

/-- **The characters are orthogonal**: `Σ_{j<n} e^{2πijd/n}` is `n` for
`d = 0` and `0` otherwise, a geometric sum of ratio `e^{2πid/n}`. -/
theorem sum_exp_characterPhase (d : Fin n) :
    ∑ j : Fin n, Complex.exp (↑((characterPhase n (j : ℕ)).ofLp d) * Complex.I) =
      if d = 0 then (n : ℂ) else 0 := by
  have hn : (0 : ℝ) < n := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne n)
  have hnC : (n : ℂ) ≠ 0 := Nat.cast_ne_zero.2 (NeZero.ne n)
  set q := Complex.exp (↑(2 * π * ((d : ℕ) : ℝ) / n) * Complex.I) with hq
  have hpow : ∀ j : ℕ, Complex.exp (↑((characterPhase n j).ofLp d) * Complex.I) = q ^ j := by
    intro j
    rw [hq, ← Complex.exp_nat_mul]
    congr 1
    simp only [characterPhase, WithLp.ofLp_toLp]
    push_cast
    ring
  simp only [hpow]
  rw [Fin.sum_univ_eq_sum_range (fun j => q ^ j) n]
  split_ifs with hd
  · simp [hq, hd]
  · have hd0 : (d : ℕ) ≠ 0 := fun h => hd (Fin.ext (by simp [h]))
    have hq1 : q ≠ 1 := by
      rw [hq, Ne, Complex.exp_eq_one_iff]
      rintro ⟨k, hk⟩
      have h1 := congrArg Complex.im hk
      simp at h1
      have h2 : ((d : ℕ) : ℝ) = k * n := by
        field_simp at h1
        nlinarith [pi_pos]
      have hdpos : (0 : ℝ) < (d : ℕ) := by exact_mod_cast Nat.pos_of_ne_zero hd0
      have hdn : ((d : ℕ) : ℝ) < n := by exact_mod_cast d.isLt
      have hk0 : (0 : ℤ) < k := by
        have : (0 : ℝ) < k := by nlinarith
        exact_mod_cast this
      have hk1 : k < 1 := by
        have : (k : ℝ) < 1 := by nlinarith
        exact_mod_cast this
      omega
    have hqn : q ^ n = 1 := by
      rw [hq, ← Complex.exp_nat_mul, Complex.exp_eq_one_iff]
      exact ⟨(d : ℕ), by push_cast; field_simp⟩
    rw [geom_sum_eq hq1, hqn, sub_self, zero_div]

/-- **Orthogonality of two characters' values**:
`Σ_j e^{i(2πja/n - 2πjb/n)}` is `n` for `a = b` and `0` otherwise. -/
theorem sum_exp_characterPhase_sub (a b : Fin n) :
    ∑ j : Fin n, Complex.exp (↑((characterPhase n (j : ℕ)).ofLp a -
      (characterPhase n (j : ℕ)).ofLp b) * Complex.I) = if a = b then (n : ℂ) else 0 := by
  have h : ∀ j : Fin n, Complex.exp (↑((characterPhase n (j : ℕ)).ofLp a -
      (characterPhase n (j : ℕ)).ofLp b) * Complex.I) =
      Complex.exp (↑((characterPhase n (j : ℕ)).ofLp (a - b)) * Complex.I) := fun j => by
    obtain ⟨k, hk⟩ := exists_characterPhase_sub ((j : ℕ) : ℤ) a b
    rw [hk, exp_ofReal_add_int_mul_two_pi_mul_I]
  simp only [h, sum_exp_characterPhase, sub_eq_zero]

/-- **Orthogonality for the relations of division**:
`Σ_j e^{i(2πjb/n + 2πjc/n - 2πja/n)}` is `n` for `a = b + c` and `0`
otherwise. -/
theorem sum_exp_characterPhase_add_sub (a b c : Fin n) :
    ∑ j : Fin n, Complex.exp (↑((characterPhase n (j : ℕ)).ofLp b +
      (characterPhase n (j : ℕ)).ofLp c - (characterPhase n (j : ℕ)).ofLp a) * Complex.I) =
      if a = b + c then (n : ℂ) else 0 := by
  have h : ∀ j : Fin n, Complex.exp (↑((characterPhase n (j : ℕ)).ofLp b +
      (characterPhase n (j : ℕ)).ofLp c - (characterPhase n (j : ℕ)).ofLp a) * Complex.I) =
      Complex.exp (↑((characterPhase n (j : ℕ)).ofLp b -
        (characterPhase n (j : ℕ)).ofLp (a - c)) * Complex.I) := fun j => by
    obtain ⟨k, hk⟩ := exists_characterPhase_sub ((j : ℕ) : ℤ) a c
    have e : (characterPhase n (j : ℕ)).ofLp b + (characterPhase n (j : ℕ)).ofLp c -
        (characterPhase n (j : ℕ)).ofLp a = (characterPhase n (j : ℕ)).ofLp b -
          (characterPhase n (j : ℕ)).ofLp (a - c) + ((-k : ℤ) : ℝ) * (2 * π) := by
      push_cast
      linarith
    rw [e, exp_ofReal_add_int_mul_two_pi_mul_I]
  simp only [h, sum_exp_characterPhase_sub, eq_sub_iff_add_eq, @eq_comm _ (b + c) a]

/-- The hypothesis `NeZero n` of the last three theorems holds for `n = 2`,
where the two characters take the values `1` and `-1` at `1`. -/
example : ∑ j : Fin 2, Complex.exp (↑((characterPhase 2 (j : ℕ)).ofLp 1) * Complex.I) = 0 := by
  rw [sum_exp_characterPhase]
  simp

end Transformer.PhaseModel
