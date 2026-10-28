module

public import Mathlib.Algebra.Polynomial.HasseDeriv
public import Mathlib.Algebra.Polynomial.Taylor
public import Mathlib.FieldTheory.Finite.Basic
public import Mathlib.RingTheory.Coprime.Lemmas

/-!
# Hasse derivatives for Stepanov's method

Stepanov's method bounds the number of points of a curve over a finite field `F` with an
auxiliary polynomial that vanishes to high order at each of them. In characteristic `p` the
derivatives `D^k` vanish for `k ≥ p`, so the method uses the Hasse derivatives
`E^{(k)} = D^k / k!` (`Polynomial.hasseDeriv`). This file proves what it needs of them:
`h^{N − i}` divides `E^{(i)}(h^N)`; for `k < q = |F|` the factor `X^{q j}` passes through
`E^{(k)}`, since `binom(q j, i) = 0` in `F` for `0 < i < q`; and a nonzero polynomial whose
Hasse derivatives of order `< ℓ` vanish at every point of `S` has degree at least `ℓ |S|`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

section CommSemiring

variable {R : Type*} [CommSemiring R]

/-- `h^{N − i}` divides `E^{(i)}(h^N)`: by Leibniz, each term of `E^{(i)}(h · h^N)` keeps a
factor `h^{N + 1 − i}`. -/
theorem pow_sub_dvd_hasseDeriv_pow (h : R[X]) (N i : ℕ) :
    h ^ (N - i) ∣ hasseDeriv i (h ^ N) := by
  induction N generalizing i with
  | zero => simp
  | succ N ih =>
    rw [pow_succ', hasseDeriv_mul]
    refine Finset.dvd_sum fun ab hab => ?_
    rw [Finset.mem_antidiagonal] at hab
    rcases Nat.eq_zero_or_pos ab.1 with h0 | h0
    · have hb : ab.2 = i := by omega
      rw [h0, hb, hasseDeriv_zero, LinearMap.id_apply]
      calc h ^ (N + 1 - i) ∣ h ^ (N - i + 1) := pow_dvd_pow h (by omega)
        _ = h * h ^ (N - i) := pow_succ' h _
        _ ∣ h * hasseDeriv i (h ^ N) := mul_dvd_mul_left h (ih i)
    · exact ((pow_dvd_pow h (by omega)).trans (ih ab.2)).mul_left _

example := pow_sub_dvd_hasseDeriv_pow (X : ℤ[X]) 3 1

end CommSemiring

section Field

variable {F : Type*} [Field F] [Fintype F]

/-- `binom(q j, i) = 0` in `F` for `0 < i < q = |F|`: in `F[X]`,
`(X + 1)^{q j} = expand_q ((X + 1)^j)` has no term `X^i` with `q ∤ i`. -/
theorem choose_card_mul_eq_zero (j : ℕ) {i : ℕ} (hi0 : 0 < i) (hiq : i < Fintype.card F) :
    ((Fintype.card F * j).choose i : F) = 0 := by
  have hq : 0 < Fintype.card F := Fintype.card_pos
  have h : ((X + 1 : F[X]) ^ (Fintype.card F * j)) =
      expand F (Fintype.card F) ((X + 1) ^ j) := by
    rw [map_pow, FiniteField.Polynomial.expand_card, pow_mul]
  have := congrArg (coeff · i) h
  simp only [coeff_X_add_one_pow, coeff_expand hq] at this
  rw [this, ite_eq_right (Nat.not_dvd_of_pos_of_lt hi0 hiq)]

example := choose_card_mul_eq_zero (F := ZMod 3) 2 (i := 2) (by decide) (by simp)

/-- For `k < q = |F|`, `E^{(k)}(h X^{q j}) = E^{(k)}(h) X^{q j}`. -/
theorem hasseDeriv_mul_X_pow_card_mul (h : F[X]) (j : ℕ) {k : ℕ} (hk : k < Fintype.card F) :
    hasseDeriv k (h * X ^ (Fintype.card F * j)) = hasseDeriv k h * X ^ (Fintype.card F * j) := by
  rw [hasseDeriv_mul, Finset.sum_eq_single (k, 0)]
  · simp
  · rintro ⟨a, b⟩ hab hne
    rw [Finset.mem_antidiagonal] at hab
    have hb : 0 < b := Nat.pos_of_ne_zero fun hb => hne (by simp [hb, ← hab])
    rw [X_pow_eq_monomial, hasseDeriv_monomial, choose_card_mul_eq_zero j hb (by omega)]
    simp
  · simp

example := hasseDeriv_mul_X_pow_card_mul (F := ZMod 3) X 2 (k := 1) (by simp)

end Field

section Roots

variable {F : Type*} [Field F]

/-- If `E^{(k)} R` vanishes at `x` for every `k < ℓ`, then `(X − x)^ℓ` divides `R`: the Taylor
expansion of `R` at `x` has the coefficients `E^{(k)} R (x)`. -/
theorem X_sub_C_pow_dvd_of_hasseDeriv {R : F[X]} {x : F} {ℓ : ℕ}
    (h : ∀ k < ℓ, (hasseDeriv k R).eval x = 0) : (X - C x) ^ ℓ ∣ R := by
  obtain ⟨Q, hQ⟩ : X ^ ℓ ∣ taylor x R :=
    X_pow_dvd_iff.mpr fun d hd => by rw [taylor_coeff]; exact h d hd
  refine ⟨taylor (-x) Q, ?_⟩
  have h1 : taylor (-x) (taylor x R) = R := by rw [taylor_taylor, neg_add_cancel, taylor_zero]
  rw [← h1, hQ, taylor_mul, taylor_X_pow, C_neg, ← sub_eq_add_neg]

example := X_sub_C_pow_dvd_of_hasseDeriv (F := ZMod 3) (R := X) (x := 0) (ℓ := 1)
  fun k hk => by rw [Nat.lt_one_iff.mp hk]; simp

/-- A nonzero `R` whose Hasse derivatives of order `< ℓ` vanish on `S` has `ℓ |S| ≤ deg R`. -/
theorem mul_card_le_natDegree [DecidableEq F] {R : F[X]} (hR : R ≠ 0) (S : Finset F) {ℓ : ℕ}
    (h : ∀ x ∈ S, ∀ k < ℓ, (hasseDeriv k R).eval x = 0) : ℓ * #S ≤ R.natDegree := by
  have hdvd : ∏ x ∈ S, (X - C x) ^ ℓ ∣ R :=
    Finset.prod_dvd_of_coprime
      (fun a _ b _ hab => ((pairwise_coprime_X_sub_C Function.injective_id) hab).pow)
      fun x hx => X_sub_C_pow_dvd_of_hasseDeriv (h x hx)
  have := natDegree_le_of_dvd hdvd hR
  rwa [natDegree_prod _ _ fun x _ => pow_ne_zero _ (X_sub_C_ne_zero x),
    Finset.sum_congr rfl fun x _ => by rw [natDegree_pow, natDegree_X_sub_C, mul_one],
    Finset.sum_const, smul_eq_mul, mul_comm] at this

example := mul_card_le_natDegree (F := ZMod 3) (R := X) X_ne_zero {0} (ℓ := 1)
  fun x hx k hk => by rw [Finset.mem_singleton.mp hx, Nat.lt_one_iff.mp hk]; simp

end Roots

end Transformer.Weil
