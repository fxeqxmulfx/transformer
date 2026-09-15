/-
# Causal attention — The interaction window (§B of 2411.04990v2)

* `Lemma lem:interaction`  — the two inequalities on the interaction scale
  `β^{-1/2}` that the hypotheses of `thm: fixed_centers` ask of `h` and `g`;
* `Remark rem:interaction` — the explicit `ε, c, β, N` window in which those
  inequalities hold, so that only `β ≳ log N` is needed.

The functions themselves, and what is known about them, are in
`Causal.Interaction`.  Neither statement is proved here.
-/

import Transformer.Basic
import Transformer.Causal.Interaction

open scoped BigOperators
open Real

namespace Transformer
namespace Causal

open Causal

/-- **Lemma (lem:interaction).** *The interaction inequalities of
`thm: fixed_centers`.*

If

  `c ≥ 5.5 + 2ε`,  `β ≥ (c - 1 - 2ε)² / 2`,  `N < e^{3(c-1-2ε)²/8} · ε/(c-1)`,

then

  `N h((c - 1 - 2ε) β^{-1/2}) < h(ε β^{-1/2})`  and
  `-N g((c - 2ε) β^{-1/2}) < g(ε β^{-1/2})`.

The second inequality is stated as the lemma states it, with the minus sign;
`thm: fixed_centers` asks for `N g((c-2ε)β^{-1/2}) < g(εβ^{-1/2})` without it,
and the two agree exactly when `g((c-2ε)β^{-1/2}) ≤ 0`, which the peak
location of `h_pot_unimodal` gives at `c - 2ε > 1`.

Not proved here.

Source: arXiv:2411.04990v2, §B, `lem:interaction`. -/
theorem interaction_inequalities
    (N c ε β : ℝ) (hε : 0 < ε) (hN : 0 < N)
    (hc : 5.5 + 2 * ε ≤ c) (hβ : (c - 1 - 2 * ε) ^ 2 / 2 ≤ β)
    (hNbound : N < Real.exp (3 * (c - 1 - 2 * ε) ^ 2 / 8) * (ε / (c - 1))) :
    N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ)))
        < h_pot β (ε * β ^ (-(1 / 2 : ℝ))) ∧
      -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ))))
        < g_pot β (ε * β ^ (-(1 / 2 : ℝ))) := by
  sorry

/-- The hypotheses of `interaction_inequalities` are satisfiable: the remark's
own example `ε = 0.1`, `c = 6.5`, `β = 14`, `N = 1`.  Note `5.5 + 2ε = 5.7 ≤
6.5`, `(c - 1 - 2ε)²/2 = 5.4²/2 = 14.58`, so `β` is taken at the remark's
rounded threshold `β ≥ 14` only after the exact one — `β = 15` clears both. -/
example :
    (0 : ℝ) < 0.1 ∧ (0 : ℝ) < 1 ∧ (5.5 : ℝ) + 2 * 0.1 ≤ 6.5 ∧
      ((6.5 : ℝ) - 1 - 2 * 0.1) ^ 2 / 2 ≤ 15 ∧
      (1 : ℝ) < Real.exp (3 * ((6.5 : ℝ) - 1 - 2 * 0.1) ^ 2 / 8) * (0.1 / (6.5 - 1)) := by
  refine ⟨by norm_num, one_pos, by norm_num, by norm_num, ?_⟩
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
      ≤ Real.exp (3 * ((6.5 : ℝ) - 1 - 2 * 0.1) ^ 2 / 8) :=
    Real.exp_le_exp.mpr (by norm_num)
  nlinarith

/-- **Remark (rem:interaction).** *An explicit window for `thm: fixed_centers`.*

The hypotheses of `thm: fixed_centers` hold whenever

  `ε < 0.1`,  `c ≥ 5.5 + 2ε`,  `β ≥ (c - 1 - 2ε)²/2`,
  `N ≤ (ε / (c - 1)) e^{3(c-1-2ε)²/8}`,

so that only `β ≳ log N` is needed.  This is `interaction_inequalities` read
as a sufficient condition, with the strict bound on `N` relaxed to `≤`; the
remark's own example is `ε = 0.1`, `c = 6.5`, `β ≥ 14`, `N ≤ 700`.

A `Prop`-valued definition: it asserts that the window implies the two
inequalities, and is not proved here.

Source: arXiv:2411.04990v2, §B, `rem:interaction`. -/
def InteractionWindow : Prop :=
  ∀ N c ε β : ℝ, 0 < ε → ε < 0.1 → 0 < N →
    5.5 + 2 * ε ≤ c → (c - 1 - 2 * ε) ^ 2 / 2 ≤ β →
    N ≤ (ε / (c - 1)) * Real.exp (3 * (c - 1 - 2 * ε) ^ 2 / 8) →
      N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ)))
          < h_pot β (ε * β ^ (-(1 / 2 : ℝ))) ∧
        N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))
          < g_pot β (ε * β ^ (-(1 / 2 : ℝ)))

end Causal
end Transformer
