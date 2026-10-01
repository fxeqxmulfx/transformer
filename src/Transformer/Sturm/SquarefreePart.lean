/-
# Removing multiplicities by division by the derivative gcd

The construction preserves every actual real root and gives it
multiplicity one. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2.
-/

import Transformer.Sturm.EuclideanChain
import Mathlib.Algebra.Polynomial.Roots

noncomputable section
open Polynomial

namespace Transformer.Sturm

/-- The computable algebraic expression `p / gcd(p, p')`, using field
division. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def squarefreePart (p : Polynomial ℝ) : Polynomial ℝ :=
  p / EuclideanDomain.gcd p p.derivative

/-- The derivative gcd of a nonzero polynomial is nonzero.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem derivative_gcd_ne_zero {p : Polynomial ℝ} (hp : p ≠ 0) :
    EuclideanDomain.gcd p p.derivative ≠ 0 := by
  intro hg
  have hd := EuclideanDomain.gcd_dvd_left p p.derivative
  rw [hg, zero_dvd_iff] at hd
  exact hp hd

/-- Division by the derivative gcd is exact.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem squarefreePart_factor {p : Polynomial ℝ} (hp : p ≠ 0) :
    EuclideanDomain.gcd p p.derivative * squarefreePart p = p :=
  EuclideanDomain.mul_div_cancel' (derivative_gcd_ne_zero hp)
    (EuclideanDomain.gcd_dvd_left p p.derivative)

/-- Removing multiplicities does not produce the zero polynomial.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem squarefreePart_ne_zero {p : Polynomial ℝ} (hp : p ≠ 0) :
    squarefreePart p ≠ 0 := by
  intro hs
  have hf := squarefreePart_factor hp
  rw [hs, mul_zero] at hf
  exact hp hf.symm

/-- A nonzero polynomial with a real root has nonzero derivative as a
polynomial; the derivative need not be nonzero at that root.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem derivative_ne_zero_of_real_root {p : Polynomial ℝ} (hp : p ≠ 0)
    {r : ℝ} (hr : p.eval r = 0) : p.derivative ≠ 0 := by
  intro hd
  have heq := eq_C_of_derivative_eq_zero hd
  have hc : p.coeff 0 = 0 := by
    have hv := congrArg (fun f : Polynomial ℝ => f.eval r) heq
    rw [hr, eval_C] at hv
    exact hv.symm
  exact hp (heq.trans (by rw [hc, C_0]))

/-- At a real root of multiplicity `m`, the derivative gcd has
multiplicity exactly `m - 1`. Auxiliary for Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
theorem derivative_gcd_rootMultiplicity {p : Polynomial ℝ} (hp : p ≠ 0)
    {r : ℝ} (hr : p.eval r = 0) :
    (EuclideanDomain.gcd p p.derivative).rootMultiplicity r = p.rootMultiplicity r - 1 := by
  have hd := derivative_ne_zero_of_real_root hp hr
  have hgd := derivative_gcd_ne_zero hp
  have hmult := derivative_rootMultiplicity_of_root (p := p) hr
  apply le_antisymm
  · simpa only [hmult] using rootMultiplicity_le_rootMultiplicity_of_dvd hd
      (EuclideanDomain.gcd_dvd_right p p.derivative) r
  · apply (le_rootMultiplicity_iff hgd).mpr
    apply EuclideanDomain.dvd_gcd
    · exact (le_rootMultiplicity_iff hp).mp (Nat.sub_le _ _)
    · exact (le_rootMultiplicity_iff hd).mp (by rw [hmult])

/-- Every real root remains with multiplicity exactly one after the
derivative gcd is removed. Auxiliary for Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
theorem squarefreePart_rootMultiplicity {p : Polynomial ℝ} (hp : p ≠ 0)
    {r : ℝ} (hr : p.eval r = 0) : (squarefreePart p).rootMultiplicity r = 1 := by
  have hf := squarefreePart_factor hp
  have hprod : EuclideanDomain.gcd p p.derivative * squarefreePart p ≠ 0 := by
    rw [hf]
    exact hp
  have hm := rootMultiplicity_mul (x := r) hprod
  rw [hf, derivative_gcd_rootMultiplicity hp hr] at hm
  have hpos : 0 < p.rootMultiplicity r := (rootMultiplicity_pos hp).mpr hr
  omega

/-- Division by the derivative gcd preserves the real zero set in both
directions, including roots of arbitrary multiplicity. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem squarefreePart_real_roots {p : Polynomial ℝ} (hp : p ≠ 0) (r : ℝ) :
    (squarefreePart p).eval r = 0 ↔ p.eval r = 0 := by
  constructor
  · intro hr
    have hf := congrArg (fun f : Polynomial ℝ => f.eval r) (squarefreePart_factor hp)
    simpa only [eval_mul, hr, mul_zero] using hf.symm
  · intro hr
    apply (rootMultiplicity_pos (squarefreePart_ne_zero hp)).mp
    rw [squarefreePart_rootMultiplicity hp hr]
    exact Nat.zero_lt_one

/-- A genuinely repeated real root witnesses the nonzero and root
assumptions used throughout multiplicity removal. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : (X ^ 4 : Polynomial ℝ) ≠ 0 ∧ (X ^ 4 : Polynomial ℝ).eval 0 = 0 := by
  exact ⟨pow_ne_zero 4 X_ne_zero, by simp⟩

end Transformer.Sturm
