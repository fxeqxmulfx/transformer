/-
# §6 — The exceptional set of the high-dimensional statements

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

`thm: boumal` and `thm: d.infty` speak of *Lebesgue-almost every* initial
sequence, and the quantifier is not decoration.  The antipodal pair `(x, -x)`
is a stationary point of `eq: SA` at every `β` (`SA_const_antipodalPair`), so
it converges to no single `x⋆` (`antipodalPair_not_mem_clusteringSet`) and a
fortiori obeys no exponential rate — which is what this file proves.  Both
statements of `Perspective.Section5_HighD` are therefore read against the
uniform law of §4, the way `thm: beta.interval` already is.
-/

import Transformer.Perspective.Section3_SmallBeta

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d : ℕ)

/-- **The antipodal pair obeys no exponential rate, at any `β`.**

The constant path through `(x, -x)` solves `eq: SA`, and its two particles stay
at distance `2`, so `‖x_i(t) - x⋆‖ ≤ C e^{-λ t}` for both of them would force
`2 ≤ 2 C e^{-λ t}` for every `t ≥ 0`, which fails at `t = (log C + 1)/λ`.

This is the exceptional set of `eq: expconvtocons`: `d_infty_exponential`
cannot be read for every initial sequence, only for almost every one.

Source: arXiv:2312.10794v5, §6.1, `thm: d.infty` (the "almost surely" of the
statement). -/
theorem antipodalPair_not_exponential (β : ℝ) (x : SSphere d) :
    ¬ ∃ (x_star : SSphere d) (C lam : ℝ),
        0 < C ∧ 0 < lam ∧
        ∀ X : ℝ → SphereTuple d 2, X 0 = antipodalPair d x →
          Perspective.SA d 2 β X →
            ∀ i : Idx 2, ∀ t : ℝ, 0 ≤ t →
              ‖((X t i : EucSpace d)) - (x_star : EucSpace d)‖
                ≤ C * Real.exp (-(lam * t)) := by
  rintro ⟨x_star, C, lam, hC, hlam, h⟩
  have hbound := h (fun _ => antipodalPair d x) rfl (SA_const_antipodalPair d β x)
  set t : ℝ := max 0 ((Real.log C + 1) / lam) with ht
  have ht0 : 0 ≤ t := le_max_left _ _
  have h0 := hbound 0 t ht0
  have h1 := hbound 1 t ht0
  have hpair0 : ((antipodalPair d x 0 : SSphere d) : EucSpace d) = (x : EucSpace d) := rfl
  have hpair1 : ((antipodalPair d x 1 : SSphere d) : EucSpace d) = -(x : EucSpace d) := rfl
  simp only [hpair0, hpair1] at h0 h1
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hgap : ‖(x : EucSpace d) - -(x : EucSpace d)‖ = 2 := by
    have hsplit : (x : EucSpace d) - -(x : EucSpace d) = (2 : ℝ) • (x : EucSpace d) := by
      module
    rw [hsplit, norm_smul, hx]
    norm_num
  have htri : ‖(x : EucSpace d) - -(x : EucSpace d)‖
      ≤ ‖(x : EucSpace d) - (x_star : EucSpace d)‖
        + ‖-(x : EucSpace d) - (x_star : EucSpace d)‖ := by
    have hd := dist_triangle ((x : EucSpace d)) ((x_star : EucSpace d)) (-(x : EucSpace d))
    rwa [dist_eq_norm, dist_eq_norm, dist_eq_norm, norm_sub_rev ((x_star : EucSpace d))] at hd
  have hdiv : Real.log C + 1 ≤ lam * t := by
    have hle := mul_le_mul_of_nonneg_left (le_max_right (0 : ℝ) ((Real.log C + 1) / lam)) hlam.le
    rwa [mul_div_cancel₀ _ (ne_of_gt hlam)] at hle
  have hstep : C * Real.exp (-(lam * t)) ≤ C * Real.exp (-(Real.log C + 1)) :=
    mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (by linarith)) hC.le
  have hval : C * Real.exp (-(Real.log C + 1)) = Real.exp (-1) := by
    rw [show -(Real.log C + 1) = -Real.log C + -1 by ring, Real.exp_add, Real.exp_neg,
      Real.exp_log hC, ← mul_assoc, mul_inv_cancel₀ (ne_of_gt hC), one_mul]
  have hlt : Real.exp (-1 : ℝ) < 1 := Real.exp_lt_one_iff.mpr (by norm_num)
  rw [hgap] at htri
  linarith

end Perspective
end Transformer
