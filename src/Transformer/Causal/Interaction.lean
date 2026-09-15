/-
# Causal attention — The interaction functions (§B of 2411.04990v2)

The interaction potential of `eq: def_h` and its derivative,

  `h(x) = e^{β(cos x - 1)} sin x`,   `g(x) = h'(x)`,

and what the paper proves about them:

`Lemma lemma:interaction` — periodicity and parity, positivity on `[0, π]`,
the unimodality with peak at `τ_β^*`, and two-sided Gaussian bounds on both
`h` and `g`.  The two inequalities `thm: fixed_centers` asks of them are in
`Causal.InteractionWindow`.

Periodicity, parity, positivity and the formula for `g` are proved here; the
quantitative parts are stated and left open.
-/

import Transformer.Basic
import Transformer.Causal.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

open scoped BigOperators
open Real

namespace Transformer
namespace Causal

open Causal

/-- **The derivative of the interaction potential:**

  `g(x) = h'(x) = e^{β(cos x - 1)} (cos x - β sin² x)`.

Source: arXiv:2411.04990v2, §B, `lemma:interaction`. -/
noncomputable def g_pot (β x : ℝ) : ℝ :=
  Real.exp (β * (Real.cos x - 1)) * (Real.cos x - β * Real.sin x ^ 2)

/-- `g_pot` is the derivative of `h_pot`. -/
theorem hasDerivAt_h_pot (β x : ℝ) :
    HasDerivAt (h_pot β) (g_pot β x) x := by
  have hexp : HasDerivAt (fun y => Real.exp (β * (Real.cos y - 1)))
      (Real.exp (β * (Real.cos x - 1)) * (β * -Real.sin x)) x :=
    (((Real.hasDerivAt_cos x).sub_const 1).const_mul β).exp
  have h := hexp.mul (Real.hasDerivAt_sin x)
  have heq : g_pot β x
      = Real.exp (β * (Real.cos x - 1)) * (β * -Real.sin x) * Real.sin x
        + Real.exp (β * (Real.cos x - 1)) * Real.cos x := by
    unfold g_pot
    ring
  rw [heq]
  exact h

/-- **Lemma (lemma:interaction), 1.** *`h` is `2π`-periodic and odd, `g` is
`2π`-periodic and even.*

Source: arXiv:2411.04990v2, §B, `lemma:interaction` (1). -/
theorem h_pot_periodic (β x : ℝ) : h_pot β (x + 2 * Real.pi) = h_pot β x := by
  simp [h_pot, Real.cos_add_two_pi, Real.sin_add_two_pi]

theorem g_pot_periodic (β x : ℝ) : g_pot β (x + 2 * Real.pi) = g_pot β x := by
  simp [g_pot, Real.cos_add_two_pi, Real.sin_add_two_pi]

theorem h_pot_odd (β x : ℝ) : h_pot β (-x) = -h_pot β x := by
  simp [h_pot]

theorem g_pot_even (β x : ℝ) : g_pot β (-x) = g_pot β x := by
  simp [g_pot]

/-- **Lemma (lemma:interaction), 2.** *`h ≥ 0` on `[0, π]`.*

Source: arXiv:2411.04990v2, §B, `lemma:interaction` (2). -/
theorem h_pot_nonneg (β : ℝ) {x : ℝ} (hx : x ∈ Set.Icc 0 Real.pi) :
    0 ≤ h_pot β x :=
  mul_nonneg (le_of_lt (Real.exp_pos _)) (Real.sin_nonneg_of_mem_Icc hx)

/-- The interval of `h_pot_nonneg` is inhabited: `0 ∈ [0, π]`. -/
example : (0 : ℝ) ∈ Set.Icc 0 Real.pi :=
  ⟨le_rfl, Real.pi_pos.le⟩

/-- **Lemma (lemma:interaction), 3.** *`h` is unimodal on `[0, π]`.*

`h` increases on `[0, τ_β^*]` and decreases on `[τ_β^*, π]`, where

  `cos τ_β^* = (-1 + √(4β² + 1)) / (2β)`,

and for `β ≥ 1` the peak sits at the interaction scale,

  `(β + 1/2)^{-1/2} < τ_β^* < β^{-1/2}`.

The peak is existentially quantified rather than constructed: `τ_β^*` is the
arccosine of the displayed value, and what the statement keeps of it is what
the proof of `thm: fixed_centers` uses — it lies in `(0, π)`, it is where the
two monotonicity intervals meet, and it is of order `β^{-1/2}`.

Not proved here.

Source: arXiv:2411.04990v2, §B, `lemma:interaction` (3). -/
theorem h_pot_unimodal (β : ℝ) (hβ : 1 ≤ β) :
    ∃ τ : ℝ, 0 < τ ∧ τ < Real.pi ∧
      Real.cos τ = (-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β) ∧
      (β + 1 / 2) ^ (-(1 / 2 : ℝ)) < τ ∧ τ < β ^ (-(1 / 2 : ℝ)) ∧
      StrictMonoOn (h_pot β) (Set.Icc 0 τ) ∧
      StrictAntiOn (h_pot β) (Set.Icc τ Real.pi) := by
  sorry

/-- The hypothesis of `h_pot_unimodal` is satisfiable: `β = 1`. -/
example : (1 : ℝ) ≤ 1 := le_rfl

/-- **Lemma (lemma:interaction), 4.** *Gaussian bounds on `h`.*

For `x > 0`,

  `e^{-βx²/2} (x - x³/6) < h(x) < e^{-βx²/2 + βx⁴/24} x`.

Not proved here.

Source: arXiv:2411.04990v2, §B, `lemma:interaction` (4). -/
theorem h_pot_bounds (β : ℝ) (hβ : 0 < β) {x : ℝ} (hx : 0 < x) :
    Real.exp (-(β * x ^ 2 / 2)) * (x - x ^ 3 / 6) < h_pot β x ∧
      h_pot β x < Real.exp (-(β * x ^ 2 / 2) + β * x ^ 4 / 24) * x := by
  sorry

/-- The hypotheses of `h_pot_bounds` are satisfiable: `β = 1`, `x = 1`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 := ⟨one_pos, one_pos⟩

/-- **Lemma (lemma:interaction), 5.** *Gaussian bounds on `g`.*

  `g(x) > e^{-βx²/2} (1 - x²/2 - β x²)`   for `0 < x < (β + 1/2)^{-1/2}`,
  `g(x) > -e^{-βx²/2 + βx⁴/24} β x²`      for `x > 0`.

Not proved here.

Source: arXiv:2411.04990v2, §B, `lemma:interaction` (5). -/
theorem g_pot_lower_bounds (β : ℝ) (hβ : 0 < β) {x : ℝ} (hx : 0 < x) :
    (x < (β + 1 / 2) ^ (-(1 / 2 : ℝ)) →
        Real.exp (-(β * x ^ 2 / 2)) * (1 - x ^ 2 / 2 - β * x ^ 2) < g_pot β x) ∧
      -(Real.exp (-(β * x ^ 2 / 2) + β * x ^ 4 / 24) * (β * x ^ 2)) < g_pot β x := by
  sorry

/-- The hypotheses of `g_pot_lower_bounds` are satisfiable: `β = 1`, `x = 1`,
and the first branch is then vacuous since `1 > (3/2)^{-1/2}`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 := ⟨one_pos, one_pos⟩

end Causal
end Transformer
