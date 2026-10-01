/-
# Removing the roots where a query polynomial vanishes

The quotient by a polynomial gcd filters the roots exactly when the
dividend has simple real roots. Auxiliary for Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2.
-/

import Transformer.Sturm.SquarefreeCount

noncomputable section
open Polynomial

namespace Transformer.Sturm

/-- Removing all factors shared with a second polynomial by exact field
division. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
def coprimePart (p q : Polynomial ℝ) : Polynomial ℝ :=
  p / EuclideanDomain.gcd p q

/-- The gcd with any second polynomial is nonzero for a nonzero first
polynomial. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem gcd_ne_zero_left {p q : Polynomial ℝ} (hp : p ≠ 0) :
    EuclideanDomain.gcd p q ≠ 0 := by
  intro hg
  have hd := EuclideanDomain.gcd_dvd_left p q
  rw [hg, zero_dvd_iff] at hd
  exact hp hd

/-- The coprime-part quotient gives an exact factorization.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem coprimePart_factor {p q : Polynomial ℝ} (hp : p ≠ 0) :
    EuclideanDomain.gcd p q * coprimePart p q = p :=
  EuclideanDomain.mul_div_cancel' (gcd_ne_zero_left hp) (EuclideanDomain.gcd_dvd_left p q)

/-- The quotient of a nonzero dividend is nonzero.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem coprimePart_ne_zero {p q : Polynomial ℝ} (hp : p ≠ 0) : coprimePart p q ≠ 0 := by
  intro hc
  have hf := coprimePart_factor (q := q) hp
  rw [hc, mul_zero] at hf
  exact hp hf.symm

/-- A divisor of a polynomial with simple real roots retains that
property; this is a statement about real roots, not splitting.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem derivative_roots_of_dvd {p s : Polynomial ℝ} (hp : p ≠ 0)
    (h : ∀ x : ℝ, p.eval x = 0 → p.derivative.eval x ≠ 0) (hs : s ∣ p) :
    ∀ x : ℝ, s.eval x = 0 → s.derivative.eval x ≠ 0 := by
  have hsn : s ≠ 0 := ne_zero_of_dvd_ne_zero hp hs
  intro x hx hd
  have hgt := (one_lt_rootMultiplicity_iff_isRoot hsn).mpr ⟨hx, hd⟩
  have hle := rootMultiplicity_le_rootMultiplicity_of_dvd hp hs x
  have hbound : p.rootMultiplicity x ≤ 1 := by
    simpa only [count_roots] using
      Multiset.nodup_iff_count_le_one.mp (real_roots_nodup_of_derivative hp h) x
  omega

/-- With simple real roots, quotienting by `gcd(p, q)` retains exactly
the roots of `p` where `q` is nonzero. It also covers a zero query
polynomial. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem coprimePart_real_roots {p q : Polynomial ℝ} (hp : p ≠ 0)
    (h : ∀ x : ℝ, p.eval x = 0 → p.derivative.eval x ≠ 0) (x : ℝ) :
    (coprimePart p q).eval x = 0 ↔ p.eval x = 0 ∧ q.eval x ≠ 0 := by
  have hf := coprimePart_factor (q := q) hp
  have he := congrArg (fun f : Polynomial ℝ => f.eval x) hf
  rw [eval_mul] at he
  constructor
  · intro hc
    have hpx : p.eval x = 0 := by simpa only [hc, mul_zero] using he.symm
    refine ⟨hpx, ?_⟩
    intro hqx
    have hg : (EuclideanDomain.gcd p q).eval x = 0 := eval_gcd_eq_zero hpx hqx
    have hgpos : 0 < (EuclideanDomain.gcd p q).rootMultiplicity x :=
      (rootMultiplicity_pos (gcd_ne_zero_left hp)).mpr hg
    have hcpos : 0 < (coprimePart p q).rootMultiplicity x :=
      (rootMultiplicity_pos (coprimePart_ne_zero hp)).mpr hc
    have hprod : EuclideanDomain.gcd p q * coprimePart p q ≠ 0 := by rw [hf]; exact hp
    have hm := rootMultiplicity_mul (x := x) hprod
    rw [hf] at hm
    have hbound : p.rootMultiplicity x ≤ 1 := by
      simpa only [count_roots] using
        Multiset.nodup_iff_count_le_one.mp (real_roots_nodup_of_derivative hp h) x
    omega
  · rintro ⟨hpx, hqx⟩
    have hgne : (EuclideanDomain.gcd p q).eval x ≠ 0 := by
      intro hg
      exact hqx (root_right_of_root_gcd hg)
    rw [hpx, mul_eq_zero] at he
    exact he.resolve_left hgne

/-- The coprime part remains simple at all its real roots.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem coprimePart_derivative_roots {p q : Polynomial ℝ} (hp : p ≠ 0)
    (h : ∀ x : ℝ, p.eval x = 0 → p.derivative.eval x ≠ 0) :
    ∀ x : ℝ, (coprimePart p q).eval x = 0 → (coprimePart p q).derivative.eval x ≠ 0 :=
  derivative_roots_of_dvd hp h ⟨EuclideanDomain.gcd p q, by
    rw [mul_comm]
    exact (coprimePart_factor hp).symm⟩

/-- `X` with query `X` has a genuine shared root and satisfies the
nonzero, simple-root, and divisor hypotheses. Auxiliary for Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : (X : Polynomial ℝ) ≠ 0 ∧
    (∀ x : ℝ, (X : Polynomial ℝ).eval x = 0 → (X : Polynomial ℝ).derivative.eval x ≠ 0) ∧
    (X : Polynomial ℝ) ∣ X := by
  exact ⟨X_ne_zero, fun _ _ => by simp, dvd_rfl⟩

end Transformer.Sturm
