/-
# Signs around simple real polynomial roots

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Sturm.EuclideanSigns
import Mathlib.Analysis.Calculus.Deriv.Polynomial
import Mathlib.Analysis.Calculus.Deriv.Slope

noncomputable section
open Filter Topology Polynomial

namespace Transformer.Sturm

/-- A zero with positive derivative is negative to its left and positive
to its right. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem positive_derivative_zero_flanks {f : ℝ → ℝ} {r d : ℝ}
    (h : HasDerivAt f d r) (hzero : f r = 0) (hd : 0 < d) :
    (∀ᶠ x in 𝓝[<] r, f x < 0) ∧ (∀ᶠ x in 𝓝[>] r, 0 < f x) := by
  obtain ⟨hleft, hright⟩ := hasDerivAt_iff_tendsto_slope_left_right.mp h
  constructor
  · filter_upwards [hleft.eventually (Ioi_mem_nhds hd), self_mem_nhdsWithin] with x hs hx
    have hxr : x - r < 0 := sub_neg.mpr hx
    rw [slope_def_field, hzero, sub_zero, div_pos_iff] at hs
    exact hs.elim (fun h => (not_lt_of_ge h.2.le hxr).elim) And.left
  · filter_upwards [hright.eventually (Ioi_mem_nhds hd), self_mem_nhdsWithin] with x hs hx
    have hxr : 0 < x - r := sub_pos.mpr hx
    rw [slope_def_field, hzero, sub_zero, div_pos_iff] at hs
    exact hs.elim And.left (fun h => (not_lt_of_ge hxr.le h.2).elim)

/-- At a simple real root, the product with the derivative crosses from
negative to positive, regardless of the sign of that derivative.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem polynomial_derivative_root_flanks {p : Polynomial ℝ} {r : ℝ}
    (hr : p.eval r = 0) (hderiv : p.derivative.eval r ≠ 0) :
    (∀ᶠ x in 𝓝[<] r, (p * p.derivative).eval x < 0) ∧
      (∀ᶠ x in 𝓝[>] r, 0 < (p * p.derivative).eval x) := by
  have hD : HasDerivAt (fun x => (p * p.derivative).eval x)
      (p.derivative.eval r ^ 2) r := by
    simpa only [derivative_mul, eval_add, eval_mul, hr, zero_mul, add_zero, pow_two]
      using (p * p.derivative).hasDerivAt r
  exact positive_derivative_zero_flanks hD (by simp only [eval_mul, hr, zero_mul])
    (sq_pos_of_ne_zero hderiv)

/-- `X` at zero jointly witnesses the positive derivative and simple
polynomial root assumptions. Auxiliary for Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
example : HasDerivAt (fun x : ℝ => x) 1 0 ∧
    (fun x : ℝ => x) 0 = 0 ∧ (0 : ℝ) < 1 ∧
    (X : Polynomial ℝ).eval 0 = 0 ∧ (X : Polynomial ℝ).derivative.eval 0 ≠ 0 := by
  exact ⟨hasDerivAt_id 0, rfl, zero_lt_one, by simp, by simp⟩

end Transformer.Sturm
