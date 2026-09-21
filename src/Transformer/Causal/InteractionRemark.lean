/-
# Causal attention — The interaction window (§B of 2411.04990v2)

`Remark rem:interaction`: the explicit `ε, c, β, N` window in which the two
interaction inequalities of `lem:interaction` hold, so that only `β ≳ log N`
is needed.

`thm: fixed_centers` prints its second condition as
`N g((c-2ε)β^{-1/2}) < g(εβ^{-1/2})`, where the lemma gives
`-(N g(...)) < g(...)`.  They do not agree: `g_pot_nonpos_core` shows that
`g` is negative at the far point, so the printed form holds for free and the
lemma's is the one with content — and the one the theorem's proof uses.
`Causal.fixed_centers` is stated with the minus sign, and the window here
delivers it.

Source: arXiv:2411.04990v2, §B, `rem:interaction`.
-/

import Transformer.Causal.InteractionWindow

open scoped BigOperators
open Real

namespace Transformer
namespace Causal

open Causal

/-- **The sign of `g` at the far point.**  In the variables of
`interaction_inequalities_core` the far point `(b+1)s` satisfies
`β ((b+1)s)² = (b+1)² ≥ 5` and `(b+1)s ≤ 1.73 < π`, so `g_pot_nonpos`
applies: the lemma's `-(N g((b+1)s))` and the printed `N g((b+1)s)` of
`thm: fixed_centers` sit on opposite sides of `0`, so the printed condition
is automatic.

Source: arXiv:2411.04990v2, §B, `lemma:interaction` (3). -/
theorem g_pot_nonpos_core (β b s : ℝ) (hb : 9 / 2 ≤ b) (hβ : b ^ 2 / 2 ≤ β)
    (hs : 0 < s) (hs2 : β * s ^ 2 = 1) : g_pot β ((b + 1) * s) ≤ 0 := by
  have hbpos : (0 : ℝ) < b := by linarith
  have hβpos : (0 : ℝ) < β := by nlinarith
  have hx : (0 : ℝ) < (b + 1) * s := mul_pos (by linarith) hs
  have hxsq : β * ((b + 1) * s) ^ 2 = (b + 1) ^ 2 := by
    have h : β * ((b + 1) * s) ^ 2 = (b + 1) ^ 2 * (β * s ^ 2) := by ring
    rw [h, hs2, mul_one]
  have hb1b : (b + 1) ^ 2 ≤ 121 / 81 * b ^ 2 := by
    nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ b - 9 / 2)
      (by linarith : (0 : ℝ) ≤ 40 * b + 18)]
  have hbig : ((b + 1) * s) ^ 2 * b ^ 2 ≤ 242 / 81 * b ^ 2 := by
    nlinarith [sq_nonneg ((b + 1) * s)]
  have hsmall : ((b + 1) * s) ^ 2 ≤ 242 / 81 := by
    have hb2 : (0 : ℝ) < b ^ 2 := by positivity
    nlinarith
  refine g_pot_nonpos β _ hβpos hx ?_ ?_
  · nlinarith [Real.pi_gt_three]
  · rw [hxsq]; nlinarith

/-- The hypotheses of `g_pot_nonpos_core` are satisfiable: `b = 9/2`,
`β = 16`, `s = 1/4`. -/
example : (9 : ℝ) / 2 ≤ 9 / 2 ∧ ((9 : ℝ) / 2) ^ 2 / 2 ≤ 16 ∧ (0 : ℝ) < 1 / 4 ∧
    (16 : ℝ) * (1 / 4) ^ 2 = 1 :=
  ⟨le_rfl, by norm_num, by norm_num, by norm_num⟩

/-- **Remark (rem:interaction).** *An explicit window for `thm: fixed_centers`.*

The hypotheses of `thm: fixed_centers` hold whenever

  `ε < 0.1`,  `c ≥ 5.5 + 2ε`,  `β ≥ (c - 1 - 2ε)²/2`,
  `N ≤ (ε / (c - 1)) e^{3(c-1-2ε)²/8}`,

so that only `β ≳ log N` is needed; the remark's own example is `ε = 0.1`,
`c = 6.5`, `β ≥ 14`, `N ≤ 700`.

This is `lem:interaction` read as a sufficient condition, and it differs from
`interaction_inequalities` only in the bound on `N`, which is the non-strict
one: `interaction_inequalities_core` takes it that way, both of its chains
being strict already at their first step.  The second inequality carries the
minus sign, as in `Causal.fixed_centers`, which corrects the printed sign.

Source: arXiv:2411.04990v2, §B, `rem:interaction`. -/
theorem interaction_window (N c ε β : ℝ) (hε : 0 < ε) (hsmall : ε < 0.1) (hN : 0 < N)
    (hc : 5.5 + 2 * ε ≤ c) (hβ : (c - 1 - 2 * ε) ^ 2 / 2 ≤ β)
    (hNbound : N ≤ (ε / (c - 1)) * Real.exp (3 * (c - 1 - 2 * ε) ^ 2 / 8)) :
    N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ)))
        < h_pot β (ε * β ^ (-(1 / 2 : ℝ))) ∧
      -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ))))
        < g_pot β (ε * β ^ (-(1 / 2 : ℝ))) := by
  have hb : 9 / 2 ≤ c - 1 - 2 * ε := by norm_num at hc ⊢; linarith
  have hsmall' : ε < 1 / 10 := by norm_num at hsmall ⊢; linarith
  have hβpos : (0 : ℝ) < β := by nlinarith
  have hs : (0 : ℝ) < β ^ (-(1 / 2 : ℝ)) := Real.rpow_pos_of_pos hβpos _
  have hs2 : β * (β ^ (-(1 / 2 : ℝ))) ^ 2 = 1 := by
    have h2 : (β ^ (-(1 / 2 : ℝ))) ^ 2 = β⁻¹ := by
      rw [← Real.rpow_natCast (β ^ (-(1 / 2 : ℝ))) 2, ← Real.rpow_mul hβpos.le]
      norm_num
      rw [Real.rpow_neg_one]
    rw [h2, mul_inv_cancel₀ (ne_of_gt hβpos)]
  have hNbound' : N ≤ Real.exp (3 * (c - 1 - 2 * ε) ^ 2 / 8)
      * (ε / (c - 1 - 2 * ε + 2 * ε)) := by
    rw [show c - 1 - 2 * ε + 2 * ε = c - 1 by ring, mul_comm]; exact hNbound
  rw [show c - 2 * ε = c - 1 - 2 * ε + 1 by ring]
  exact interaction_inequalities_core N ε β _ _ hε hsmall' hN hb hβ hs hs2 hNbound'

/-- The hypotheses of `interaction_window` are satisfiable: the same example as
for `interaction_inequalities`, `ε = 0.05`, `c = 6.5`, `β = 15`, `N = 1`. -/
example :
    (0 : ℝ) < 0.05 ∧ (0.05 : ℝ) < 0.1 ∧ (0 : ℝ) < 1 ∧ (5.5 : ℝ) + 2 * 0.05 ≤ 6.5 ∧
      ((6.5 : ℝ) - 1 - 2 * 0.05) ^ 2 / 2 ≤ 15 ∧
      (1 : ℝ) ≤ (0.05 / (6.5 - 1)) * Real.exp (3 * ((6.5 : ℝ) - 1 - 2 * 0.05) ^ 2 / 8) := by
  refine ⟨by norm_num, by norm_num, one_pos, by norm_num, by norm_num, ?_⟩
  have h1 : (3.6 : ℝ) ≤ Real.exp 2.6 := by
    nlinarith [Real.add_one_le_exp (2.6 : ℝ)]
  have h2 : Real.exp 10.4 = Real.exp 2.6 ^ 4 := by
    rw [show (10.4 : ℝ) = (4 : ℕ) * 2.6 by norm_num, Real.exp_nat_mul]
  have h3 : (167 : ℝ) ≤ Real.exp 10.4 := by
    rw [h2]
    have := pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 3.6) h1 4
    norm_num at this
    linarith
  have h4 : Real.exp 10.4
      ≤ Real.exp (3 * ((6.5 : ℝ) - 1 - 2 * 0.05) ^ 2 / 8) :=
    Real.exp_le_exp.mpr (by norm_num)
  nlinarith

end Causal
end Transformer
