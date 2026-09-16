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
Gaussian bounds of items (4) and (5) are in `Causal.InteractionBounds`, and
the unimodality of item (3) — which needs the quartic bound on the cosine
proved there — is in `Causal.InteractionPeak`.
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

end Causal
end Transformer
