module

public import Mathlib.Algebra.Polynomial.FieldDivision
public import Mathlib.RingTheory.UniqueFactorizationDomain.Multiplicative
public import Mathlib.Tactic.IntervalCases
public import Mathlib.Tactic.LinearCombination

/-!
# Polynomials whose irreducible factors are all squared

Theorem B (`review/flat_spectrum.md`) for a polynomial `P` and a character `χ ≠ 1` uses Weil's
bound for `∑_a χ(c − P(a))`, which `Transformer.Weil` proves as soon as `c − P` has an
irreducible factor `R` with `R² ∤ c − P`. For `P` of degree `m ≥ 1` with `P' ≠ 0`, this holds
for every `c` but at most one:

* if `R² ∣ f` for every irreducible `R ∣ f`, then `f ∣ f'²`: a factor `R^e`, `e ≥ 2`, leaves
  `R^{e − 1}` in `f'`, and `2 (e − 1) ≥ e`;
* two such `c ≠ c'` would give `c − P` and `c' − P`, coprime as their difference is a nonzero
  constant, of total degree `2 m`, both dividing `P'²`, of degree `≤ 2 m − 2`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Polynomial

variable {K : Type*} [Field K]


/-- **A polynomial whose irreducible factors are all squared divides `f'²`**: if `R² ∣ f` for
every irreducible `R ∣ f`, then `f ∣ f'²`, as `R^e ∣ f` leaves `R^{e − 1}` in `f'`. -/
theorem dvd_derivative_sq_of_forall_sq_dvd {f : K[X]}
    (hf : ∀ R : K[X], Irreducible R → R ∣ f → R ^ 2 ∣ f) : f ∣ derivative f ^ 2 := by
  induction f using UniqueFactorizationMonoid.induction_on_coprime with
  | h0 => simp
  | h1 hu => exact hu.dvd
  | @hpr R i hR =>
    rcases Nat.lt_or_ge i 2 with hi | hi
    · interval_cases i
      · rw [pow_zero]
        exact one_dvd _
      · have h2 := (pow_dvd_pow_iff hR.ne_zero hR.not_isUnit).mp
          (hf R hR.irreducible (dvd_pow_self R one_ne_zero))
        omega
    · rw [derivative_pow]
      have h1 : R ^ i ∣ R ^ (i - 1) * R ^ (i - 1) := by
        rw [← pow_add]
        exact pow_dvd_pow R (by omega)
      exact h1.trans ⟨(C (i : K) * derivative R) ^ 2, by ring⟩
  | @hcp x y hxy hx hy =>
    have hx' : x ∣ derivative x ^ 2 := hx fun R hR hRx =>
      ((hR.coprime_iff_not_dvd.mpr fun hRy => hR.not_isUnit (hxy hRx hRy)).pow_left
        (m := 2)).dvd_of_dvd_mul_right (hf R hR (hRx.mul_right y))
    have hy' : y ∣ derivative y ^ 2 := hy fun R hR hRy =>
      ((hR.coprime_iff_not_dvd.mpr fun hRx => hR.not_isUnit (hxy hRx hRy)).pow_left
        (m := 2)).dvd_of_dvd_mul_left (hf R hR (hRy.mul_left x))
    rw [derivative_mul]
    refine (isRelPrime_iff_isCoprime.mp hxy).mul_dvd ?_ ?_
    · rw [show (derivative x * y + x * derivative y) ^ 2 = derivative x ^ 2 * y ^ 2 +
        x * (2 * derivative x * y * derivative y + x * derivative y ^ 2) by ring]
      exact dvd_add (hx'.mul_right _) (dvd_mul_right x _)
    · rw [show (derivative x * y + x * derivative y) ^ 2 = y * (derivative x ^ 2 * y +
        2 * derivative x * x * derivative y) + x ^ 2 * derivative y ^ 2 by ring]
      exact dvd_add (dvd_mul_right y _) (hy'.mul_left _)

/-- **At most one `c` with every factor of `c − P` squared**, for `P` of degree `m ≥ 1` with
`P' ≠ 0`: two would give coprime `c − P`, `c' − P` dividing `P'²`, of degree `≤ 2 m − 2`. -/
theorem eq_of_forall_sq_dvd_C_sub {P : K[X]} (hP : 0 < P.natDegree) (hP' : derivative P ≠ 0)
    {c₁ c₂ : K} (h₁ : ∀ R : K[X], Irreducible R → R ∣ C c₁ - P → R ^ 2 ∣ C c₁ - P)
    (h₂ : ∀ R : K[X], Irreducible R → R ∣ C c₂ - P → R ^ 2 ∣ C c₂ - P) : c₁ = c₂ := by
  by_contra hne
  have hd (c : K) (h : ∀ R : K[X], Irreducible R → R ∣ C c - P → R ^ 2 ∣ C c - P) :
      C c - P ∣ derivative P ^ 2 := by
    have := dvd_derivative_sq_of_forall_sq_dvd h
    rwa [derivative_sub, derivative_C, zero_sub, neg_sq] at this
  have hdeg (c : K) : (C c - P).natDegree = P.natDegree :=
    natDegree_sub_eq_right_of_natDegree_lt (by rw [natDegree_C]; exact hP)
  have hne0 (c : K) : C c - P ≠ 0 := ne_zero_of_natDegree_gt (n := 0) (by rw [hdeg]; exact hP)
  have hc : C (c₁ - c₂)⁻¹ * (C c₁ - C c₂) = 1 := by
    rw [← C_sub, ← C_mul, inv_mul_cancel₀ (sub_ne_zero.mpr hne), C_1]
  have hcop : IsCoprime (C c₁ - P) (C c₂ - P) :=
    ⟨C (c₁ - c₂)⁻¹, -C (c₁ - c₂)⁻¹, by linear_combination hc⟩
  have hle := natDegree_le_of_dvd (hcop.mul_dvd (hd c₁ h₁) (hd c₂ h₂)) (pow_ne_zero 2 hP')
  rw [natDegree_mul (hne0 c₁) (hne0 c₂), hdeg, hdeg, natDegree_pow] at hle
  have := natDegree_derivative_le P
  omega

example := dvd_derivative_sq_of_forall_sq_dvd (K := ℚ) (f := X ^ 2)
  fun _ hR h => pow_dvd_pow_of_dvd (hR.prime.dvd_of_dvd_pow h) 2

/-- For `P = X²` over `ℚ`, the one `c` with every factor of `c − P` squared is `c = 0`. -/
example := eq_of_forall_sq_dvd_C_sub (K := ℚ) (P := X ^ 2) (by rw [natDegree_X_pow]; norm_num)
  (by rw [derivative_X_pow]; simp) (c₁ := 0) (c₂ := 0)
  (fun R hR h => by
    rw [C_0, zero_sub, dvd_neg] at h ⊢
    exact pow_dvd_pow_of_dvd (hR.prime.dvd_of_dvd_pow h) 2)
  (fun R hR h => by
    rw [C_0, zero_sub, dvd_neg] at h ⊢
    exact pow_dvd_pow_of_dvd (hR.prime.dvd_of_dvd_pow h) 2)

end Transformer.Grokking.FlatSpectrum
