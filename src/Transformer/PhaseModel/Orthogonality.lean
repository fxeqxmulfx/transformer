/-
# Phase model — the characters of `ℤ/n` and their orthogonality

The phases `2πjx/n` of a character of `ℤ/n` are additive up to `2πℤ`
(`exists_characterPhase_sub`), so `x ↦ e^{2πijx/n}` is a character of
`Fin n` (`cyclicChar`), and `j ↦ cyclicChar n j` is a bijection from `Fin n`
onto the characters (`bijective_cyclicChar`).  The phases `arg e^{2πijx/n}`
of `cyclicChar n j` (`Abelian.characterPhase`) differ from `2πjx/n` by a
period (`characterPhase_sub_mem_periods`), so they give the same cosines
(`cos_sub_characterPhase`) and the same periods
(`sub_characterPhase_mem_periods_iff`).  This is how `PhaseModel.Abelian`,
at `G = Fin n`, gives the theorems on `ℤ/n`, a sum over the characters
being a sum over `j < n`.  Here it gives the orthogonality of the
characters: `Σ_j e^{i(2πja/n - 2πjb/n)}` is `n` for `a = b` and `0`
otherwise (`sum_exp_characterPhase_sub`), so `Σ_{j<n} e^{2πijd/n}` is `n`
for `d = 0` and `0` otherwise (`sum_exp_characterPhase`), and
`Σ_j e^{i(2πjb/n + 2πjc/n - 2πja/n)}` is `n` for `a = b + c` and `0`
otherwise (`sum_exp_characterPhase_add_sub`).

Source: item M6.8 of `landscape_plan.md`.
-/

import Transformer.PhaseModel.Abelian.Fourier
import Transformer.PhaseModel.Division

open Real

namespace Transformer.PhaseModel

variable {n : ℕ}

/-- **The phases of a character are additive up to `2πℤ`.** -/
theorem exists_characterPhase_sub (j : ℤ) (a b : Fin n) : ∃ k : ℤ,
    (characterPhase n j).ofLp a - (characterPhase n j).ofLp b =
      (characterPhase n j).ofLp (a - b) + k * (2 * π) := by
  obtain ⟨k, hk⟩ := exists_inner_intVec_divisionRow_characterPhase j (a, b)
  rw [inner_intVec_divisionRow] at hk
  exact ⟨k, by dsimp only at hk; linarith⟩

/-- **The character `x ↦ e^{2πijx/n}` of `ℤ/n`**, on `Fin n`: a character
as the phases `2πjx/n` are additive up to `2πℤ`. -/
noncomputable def cyclicChar (n : ℕ) [NeZero n] (j : ℤ) : AddChar (Fin n) ℂ where
  toFun x := Complex.exp (↑((characterPhase n j).ofLp x) * Complex.I)
  map_zero_eq_one' := by simp [characterPhase]
  map_add_eq_mul' a b := by
    obtain ⟨k, hk⟩ := exists_characterPhase_sub j (a + b) a
    rw [add_sub_cancel_left] at hk
    rw [show (characterPhase n j).ofLp (a + b) = (characterPhase n j).ofLp a +
      (characterPhase n j).ofLp b + k * (2 * π) by linarith, exp_ofReal_add_int_mul_two_pi_mul_I,
      ← Complex.exp_add]
    congr 1
    push_cast
    ring

variable [NeZero n]

/-- `cyclicChar n j` takes the values `e^{2πijx/n}`. -/
theorem cyclicChar_apply (j : ℤ) (x : Fin n) :
    cyclicChar n j x = Complex.exp (↑((characterPhase n j).ofLp x) * Complex.I) :=
  rfl

/-- **Distinct `j < n` give distinct characters**: their values
`e^{2πij/n}` at `1` differ. -/
theorem injective_cyclicChar : Function.Injective fun j : Fin n => cyclicChar n (j : ℕ) := by
  intro j j' h
  rcases Nat.lt_or_ge n 2 with hn | hn
  · have : Subsingleton (Fin n) := Fin.subsingleton_iff_le_one.2 (by omega)
    exact Subsingleton.elim _ _
  have h1 := DFunLike.congr_fun h ⟨1, by omega⟩
  simp only [cyclicChar_apply, characterPhase, WithLp.ofLp_toLp] at h1
  obtain ⟨k, hk⟩ := Complex.exp_eq_exp_iff_exists_int.1 h1
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (show 0 < n by omega)
  have hk' : ((j : ℕ) : ℝ) = (j' : ℕ) + k * n := by
    have h2 := congrArg Complex.im hk
    simp at h2
    field_simp at h2
    nlinarith [pi_pos]
  have hj : ((j : ℕ) : ℝ) < n := by exact_mod_cast j.isLt
  have hj' : ((j' : ℕ) : ℝ) < n := by exact_mod_cast j'.isLt
  have hk0 : k = 0 := by
    have h3 : (-1 : ℝ) < k := by
      by_contra hc
      push Not at hc
      nlinarith [(Nat.cast_nonneg (j : ℕ) : (0 : ℝ) ≤ _)]
    have h4 : (k : ℝ) < 1 := by
      by_contra hc
      push Not at hc
      nlinarith [(Nat.cast_nonneg (j' : ℕ) : (0 : ℝ) ≤ _)]
    have h5 : (-1 : ℤ) < k := by exact_mod_cast h3
    have h6 : k < 1 := by exact_mod_cast h4
    omega
  rw [hk0, Int.cast_zero, zero_mul, add_zero] at hk'
  exact Fin.ext (by exact_mod_cast hk')

/-- **The characters of `ℤ/n` are the `cyclicChar n j`, `j < n`**: `Fin n`
has `n` characters (`AddChar.card_eq`). -/
theorem bijective_cyclicChar : Function.Bijective fun j : Fin n => cyclicChar n (j : ℕ) :=
  (Fintype.bijective_iff_injective_and_card _).2 ⟨injective_cyclicChar, by rw [AddChar.card_eq]⟩

/-- **The phases `2πjx/n` are those of `cyclicChar n j` up to periods**:
both have the exponential `e^{2πijx/n}`. -/
theorem characterPhase_sub_mem_periods (j : ℤ) :
    characterPhase n j - Abelian.characterPhase (cyclicChar n j) ∈ periods n := fun x =>
  (Abelian.exp_ofReal_mul_I_eq_one_iff _).1 (by
    rw [WithLp.ofLp_sub, Pi.sub_apply, Complex.ofReal_sub, sub_mul, Complex.exp_sub,
      Abelian.exp_characterPhase, cyclicChar_apply]
    exact div_self (Complex.exp_ne_zero _))

/-- The phases `2πjx/n` and those of `cyclicChar n j` give the same
cosines. -/
theorem cos_sub_characterPhase (θ : ℝ) (j : ℤ) (x : Fin n) :
    cos (θ - (characterPhase n j).ofLp x) =
      cos (θ - (Abelian.characterPhase (cyclicChar n j)).ofLp x) := by
  obtain ⟨k, hk⟩ := characterPhase_sub_mem_periods (n := n) j x
  rw [WithLp.ofLp_sub, Pi.sub_apply] at hk
  rw [show θ - (characterPhase n j).ofLp x =
    θ - (Abelian.characterPhase (cyclicChar n j)).ofLp x - k * (2 * π) by linarith,
    cos_sub_int_mul_two_pi]

/-- The phases `ψ` are `2πjx/n` up to periods exactly when they are those
of `cyclicChar n j` up to periods. -/
theorem sub_characterPhase_mem_periods_iff (ψ : EucSpace n) (j : ℤ) :
    ψ - characterPhase n j ∈ periods n ↔
      ψ - Abelian.characterPhase (cyclicChar n j) ∈ Abelian.periods (Fin n) := by
  have hv := characterPhase_sub_mem_periods (n := n) j
  refine ⟨fun h x => ?_, fun h x => ?_⟩
  all_goals
    obtain ⟨k, hk⟩ := h x
    obtain ⟨m, hm⟩ := hv x
    simp only [WithLp.ofLp_sub, Pi.sub_apply] at hk hm ⊢
  · exact ⟨k + m, by push_cast; linarith⟩
  · exact ⟨k - m, by push_cast; linarith⟩

/-- **Orthogonality of two characters' values**:
`Σ_j e^{i(2πja/n - 2πjb/n)}` is `n` for `a = b` and `0` otherwise, the sum
over the characters of `Fin n` (`Abelian.sum_exp_characterPhase_sub`). -/
theorem sum_exp_characterPhase_sub (a b : Fin n) :
    ∑ j : Fin n, Complex.exp (↑((characterPhase n (j : ℕ)).ofLp a -
      (characterPhase n (j : ℕ)).ofLp b) * Complex.I) = if a = b then (n : ℂ) else 0 := by
  have h := Abelian.sum_exp_characterPhase_sub a b
  rw [Fintype.card_fin] at h
  refine (Finset.sum_congr rfl fun j _ => ?_).trans ((bijective_cyclicChar.sum_comp _).trans h)
  simp only [Complex.ofReal_sub, sub_mul, Complex.exp_sub, Abelian.exp_characterPhase,
    ← cyclicChar_apply]

/-- **The characters are orthogonal**: `Σ_{j<n} e^{2πijd/n}` is `n` for
`d = 0` and `0` otherwise, `sum_exp_characterPhase_sub` at `b = 0`. -/
theorem sum_exp_characterPhase (d : Fin n) :
    ∑ j : Fin n, Complex.exp (↑((characterPhase n (j : ℕ)).ofLp d) * Complex.I) =
      if d = 0 then (n : ℂ) else 0 := by
  have h0 : ∀ j : ℤ, (characterPhase n j).ofLp 0 = 0 := fun j => by simp [characterPhase]
  simpa only [h0, sub_zero] using sum_exp_characterPhase_sub d 0

/-- **Orthogonality for the relations of division**:
`Σ_j e^{i(2πjb/n + 2πjc/n - 2πja/n)}` is `n` for `a = b + c` and `0`
otherwise, the sum over the characters of `Fin n`
(`Abelian.sum_exp_characterPhase_add_sub`). -/
theorem sum_exp_characterPhase_add_sub (a b c : Fin n) :
    ∑ j : Fin n, Complex.exp (↑((characterPhase n (j : ℕ)).ofLp b +
      (characterPhase n (j : ℕ)).ofLp c - (characterPhase n (j : ℕ)).ofLp a) * Complex.I) =
      if a = b + c then (n : ℂ) else 0 := by
  have h := Abelian.sum_exp_characterPhase_add_sub a b c
  rw [Fintype.card_fin] at h
  refine (Finset.sum_congr rfl fun j _ => ?_).trans ((bijective_cyclicChar.sum_comp _).trans h)
  simp only [Complex.ofReal_sub, Complex.ofReal_add, sub_mul, add_mul, Complex.exp_sub,
    Complex.exp_add, Abelian.exp_characterPhase, ← cyclicChar_apply]

/-- The hypothesis `NeZero n` of the theorems above holds for `n = 4`: the
four characters of `ℤ/4` are `cyclicChar 4 j`, `j < 4`. -/
example : Function.Bijective fun j : Fin 4 => cyclicChar 4 (j : ℕ) :=
  bijective_cyclicChar

/-- For `n = 2` the two characters take the values `1` and `-1` at `1`. -/
example : ∑ j : Fin 2, Complex.exp (↑((characterPhase 2 (j : ℕ)).ofLp 1) * Complex.I) = 0 := by
  rw [sum_exp_characterPhase]
  simp

end Transformer.PhaseModel
