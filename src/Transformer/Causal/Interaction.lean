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
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.Real.Pi.Bounds

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

/-- **`g` is nonpositive past the interaction scale.**  On the first arch of
the sine, `g(x) ≤ 0` as soon as `β x² ≥ 5`: for `x ≥ π/2` because `cos x ≤ 0`
there, and for `x < π/2` because `sin x ≥ (2/π) x` there, so
`β sin² x ≥ (4/π²) β x² ≥ 20/π² > 1 ≥ cos x`.

The threshold is the crude one the argument gives, not a sharp one: the sign
changes at the peak `τ_β^* < β^{-1/2}` of `h_pot_unimodal`, i.e. already at
`β x² > 1`.  What it is for is `interaction_window`, which needs the sign of
`g` at `(c - 2ε) β^{-1/2}` with `c - 2ε ≥ 5.5`.

Source: arXiv:2411.04990v2, §B, `lemma:interaction` (3). -/
theorem g_pot_nonpos (β x : ℝ) (hβ : 0 < β) (hx : 0 < x) (hxπ : x ≤ Real.pi)
    (hbx : 5 ≤ β * x ^ 2) : g_pot β x ≤ 0 := by
  have hpi0 : (0 : ℝ) < Real.pi := Real.pi_pos
  have hpi : Real.pi < 4 := Real.pi_lt_four
  have key : Real.cos x - β * Real.sin x ^ 2 ≤ 0 := by
    rcases le_or_gt (Real.pi / 2) x with h | h
    · have hcos : Real.cos x ≤ 0 :=
        Real.cos_nonpos_of_pi_div_two_le_of_le h (by linarith)
      nlinarith [sq_nonneg (Real.sin x)]
    · have hlow : 2 / Real.pi * x ≤ Real.sin x := Real.mul_le_sin hx.le h.le
      have h1 : 4 * x ^ 2 / Real.pi ^ 2 ≤ Real.sin x ^ 2 := by
        have h := pow_le_pow_left₀ (by positivity : (0 : ℝ) ≤ 2 / Real.pi * x) hlow 2
        calc 4 * x ^ 2 / Real.pi ^ 2 = (2 / Real.pi * x) ^ 2 := by field_simp; ring
          _ ≤ Real.sin x ^ 2 := h
      have h2 : β * (4 * x ^ 2 / Real.pi ^ 2) ≤ β * Real.sin x ^ 2 :=
        mul_le_mul_of_nonneg_left h1 hβ.le
      have h3 : 20 / Real.pi ^ 2 ≤ β * (4 * x ^ 2 / Real.pi ^ 2) := by
        have e : β * (4 * x ^ 2 / Real.pi ^ 2) = 4 * (β * x ^ 2) / Real.pi ^ 2 := by ring
        rw [e]
        gcongr
        linarith
      have h4 : (1 : ℝ) < 20 / Real.pi ^ 2 := by
        rw [lt_div_iff₀ (by positivity)]
        nlinarith
      nlinarith [Real.cos_le_one x]
  have hexp := Real.exp_pos (β * (Real.cos x - 1))
  unfold g_pot
  nlinarith

/-- The hypotheses of `g_pot_nonpos` are satisfiable: `β = 5`, `x = 1`. -/
example : (0 : ℝ) < 5 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ Real.pi ∧ (5 : ℝ) ≤ 5 * 1 ^ 2 :=
  ⟨by norm_num, by norm_num, by linarith [Real.pi_gt_three], by norm_num⟩

end Causal
end Transformer
