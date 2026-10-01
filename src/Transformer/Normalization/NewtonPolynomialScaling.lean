/-
# Polynomial factorization after Newton scaling

The coefficient normalization applies to the entire monic polynomial, in
arbitrary degree. Away from the ramification point the original and
normalized zero sets are equivalent under the power substitutions.
-/

import Transformer.Normalization.NewtonCoefficientScaling
import Mathlib.Algebra.BigOperators.Ring.Finset

open Filter Finset
open scoped BigOperators

namespace Transformer.Normalization

/-- Weighted coefficient identities give an exact factorization of the
whole polynomial after the substitutions `t = s^q`, `y = s^p w`.
Auxiliary Newton step for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem newton_polynomial_factorization {d p q : ℕ}
    (a b : Fin d → ℝ → ℝ)
    (hcoeff : ∀ᶠ s in nhds (0 : ℝ), ∀ i,
      a i (s ^ q) = s ^ (p * (d - i.val)) * b i s) :
    ∀ᶠ s in nhds (0 : ℝ), ∀ y : ℝ,
      (s ^ p * y) ^ d + ∑ i : Fin d, a i (s ^ q) * (s ^ p * y) ^ i.val =
        s ^ (p * d) * (y ^ d + ∑ i : Fin d, b i s * y ^ i.val) := by
  filter_upwards [hcoeff] with s hs
  intro y
  rw [mul_add, mul_pow, ← pow_mul, mul_sum]
  congr 1
  apply sum_congr rfl
  intro i hi
  rw [hs i, mul_pow, ← pow_mul]
  have hexp : p * (d - i.val) + p * i.val = p * d := by
    rw [← Nat.mul_add, Nat.sub_add_cancel i.isLt.le]
  calc
    s ^ (p * (d - i.val)) * b i s * (s ^ (p * i.val) * y ^ i.val) =
        s ^ (p * (d - i.val) + p * i.val) * (b i s * y ^ i.val) := by
      rw [pow_add]
      ring
    _ = s ^ (p * d) * (b i s * y ^ i.val) := by rw [hexp]

/-- Every nontrivial analytic monic polynomial with central polynomial
`y^d` admits a Newton normalization of the full polynomial in arbitrary
degree. Its normalized central polynomial has a nonzero lower coefficient.
Identically zero coefficient germs remain identically zero coefficients.
Auxiliary for the general singular curve-lifting construction in
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_newton_polynomial_scaling {d : ℕ} (a : Fin d → ℝ → ℝ)
    (ha : ∀ i, AnalyticAt ℝ (a i) 0) (ha0 : ∀ i, a i 0 = 0)
    (hnonzero : ∃ i, ¬ ∀ᶠ t in nhds (0 : ℝ), a i t = 0) :
    ∃ (p q : ℕ) (b : Fin d → ℝ → ℝ), 0 < p ∧ 0 < q ∧
      (∀ i, AnalyticAt ℝ (b i) 0) ∧ (∃ i, b i 0 ≠ 0) ∧
      (∀ i, (∀ᶠ t in nhds (0 : ℝ), a i t = 0) → b i = fun _ => 0) ∧
      ∀ᶠ s in nhds (0 : ℝ), ∀ y : ℝ,
        (s ^ p * y) ^ d + ∑ i : Fin d, a i (s ^ q) * (s ^ p * y) ^ i.val =
          s ^ (p * d) * (y ^ d + ∑ i : Fin d, b i s * y ^ i.val) := by
  obtain ⟨p, q, b, hp, hq, hb, hb0, hzero, hcoeff⟩ :=
    analytic_newton_coefficient_scaling a ha ha0 hnonzero
  exact ⟨p, q, b, hp, hq, hb, hb0, hzero,
    newton_polynomial_factorization a b hcoeff⟩

/-- Newton scaling preserves the root set at every nonzero sufficiently
small ramified parameter. This uses the actual polynomial identity, with
the nonzero common factor cancelled. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem newton_polynomial_roots_iff {d p q : ℕ}
    (a b : Fin d → ℝ → ℝ)
    (hcoeff : ∀ᶠ s in nhds (0 : ℝ), ∀ i,
      a i (s ^ q) = s ^ (p * (d - i.val)) * b i s) :
    ∀ᶠ s in nhds (0 : ℝ), s ≠ 0 → ∀ y : ℝ,
      ((s ^ p * y) ^ d + ∑ i : Fin d, a i (s ^ q) * (s ^ p * y) ^ i.val = 0) ↔
        y ^ d + ∑ i : Fin d, b i s * y ^ i.val = 0 := by
  filter_upwards [newton_polynomial_factorization a b hcoeff] with s hs
  intro hne y
  rw [hs y, mul_eq_zero, or_iff_right (pow_ne_zero _ hne)]

/-- The cusp polynomial `y³ - t²` has the exact scaling `t = s³`,
`y = s² w`; the normalized polynomial is `w³ - 1`. Both the factorization
and root equivalence hypotheses are simultaneously satisfied. Auxiliary
example for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : (∀ᶠ s in nhds (0 : ℝ), ∀ y : ℝ,
    (s ^ 2 * y) ^ 3 - (s ^ 3) ^ 2 = s ^ 6 * (y ^ 3 - 1)) ∧
    ∀ᶠ s in nhds (0 : ℝ), s ≠ 0 → ∀ y : ℝ,
      (s ^ 2 * y) ^ 3 - (s ^ 3) ^ 2 = 0 ↔ y ^ 3 - 1 = 0 := by
  let a : Fin 3 → ℝ → ℝ := fun i t => if i = 0 then -(t ^ 2) else 0
  let b : Fin 3 → ℝ → ℝ := fun i _ => if i = 0 then -1 else 0
  have hcoeff : ∀ᶠ s in nhds (0 : ℝ), ∀ i : Fin 3,
      a i (s ^ 3) = s ^ (2 * (3 - i.val)) * b i s := by
    exact Eventually.of_forall (fun s i => by
      fin_cases i <;> simp [a, b, ← pow_mul])
  constructor
  · filter_upwards [newton_polynomial_factorization a b hcoeff] with s hs
    intro y
    simpa [a, b, Fin.sum_univ_succ, ← sub_eq_add_neg] using hs y
  · filter_upwards [newton_polynomial_roots_iff a b hcoeff] with s hs
    intro hne y
    simpa [a, b, Fin.sum_univ_succ, ← sub_eq_add_neg] using hs hne y

end Transformer.Normalization
