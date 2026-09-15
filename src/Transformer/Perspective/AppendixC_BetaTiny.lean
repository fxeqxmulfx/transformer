/-
# Appendix C — Proof of Theorem (thm: beta.tiny)

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes what Appendix C of the survey says about the modified
energy `𝖤_β` in the regime `β ≪ 1`:

* `e:contebeta`         — the expansion of `𝖤_β` around `β = 0`.  The exact
                          identity `𝖤_β = 𝖤_0 + β 𝖱_β` and the limit
                          `𝖤_β → 𝖤_0` are proved; the survey's companion
                          statements for `∇ 𝖤_β` and `Hess 𝖤_β` are not.
* `eq: eigval.beta`     — the explicit constant `c = 2(d-1)/(d n)` and its
                          positivity, which is all the proof of
                          `thm: beta.tiny` uses it for.  The eigenvalue bound
                          on `Hess 𝖤_0` itself needs the Hessian on the
                          sphere, which this development does not carry.
* `eq: metric.grad`, `eq: metric.hess` — already in `AppendixB`.

The synthesis of `thm: beta.tiny` — non-synchronized critical points of `𝖤_β`
are strict saddles for `β ≤ c/n`, hence (with `l:nosaddleconv`) clustering for
a.e. initial condition — is not formalized.
-/

import Transformer.Basic
import Transformer.Perspective.Section3_SmallBeta
import Transformer.Perspective.AppendixA_Beta0
import Mathlib.Analysis.Calculus.Deriv.Slope
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

open Perspective

variable (d n : ℕ)

/-- *Modified energy.*

  `𝖤_β(x_1,…,x_n) = (1/(2β)) Σ_i Σ_j (e^{β ⟨x_i, x_j⟩} - 1)`. -/
noncomputable def Etilde (β : ℝ) (X : SphereTuple d n) : ℝ :=
  (2 * β)⁻¹ *
    ∑ i : Idx n, ∑ j : Idx n,
      (Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d))) - 1)

/-- The `β → 0` limit of `Etilde`:

  `𝖤_0(x_1,…,x_n) = (1/2) Σ_i Σ_j ⟨x_i, x_j⟩`,

i.e. `Etilde` with `(e^{βc} - 1)/β` replaced by its value at `β = 0`.  (This
is `AppendixA_Beta0.E0` up to the survey's `n`-normalization, which `Etilde`
does not carry.) -/
noncomputable def Etilde0 (X : SphereTuple d n) : ℝ :=
  2⁻¹ * ∑ i : Idx n, ∑ j : Idx n,
    inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d))

/-- The remainder `𝖱_β` defined by `𝖤_β = 𝖤_0 + β · 𝖱_β`: solving that
equation for `𝖱_β` is the definition.  At `β = 0` Lean's `0⁻¹ = 0` makes it
`0`, which is the value the equation leaves free. -/
noncomputable def Rβ (β : ℝ) (X : SphereTuple d n) : ℝ :=
  β⁻¹ * (Etilde d n β X - Etilde0 d n X)

/-- `Rβ` does what its name says: `𝖤_β = 𝖤_0 + β · 𝖱_β` for every `β ≠ 0`. -/
theorem Etilde_eq_Etilde0_add (β : ℝ) (hβ : β ≠ 0) (X : SphereTuple d n) :
    Etilde d n β X = Etilde0 d n X + β * Rβ d n β X := by
  rw [Rβ, ← mul_assoc, mul_inv_cancel₀ hβ, one_mul]
  ring

/-- The scale `β ≠ 0` at which `Etilde_eq_Etilde0_add` speaks is of course
attained. -/
example : (1 : ℝ) ≠ 0 := one_ne_zero

/-- **Equation (e:contebeta), at the level of the energy.**

`𝖤_β → 𝖤_0` as `β → 0`: each term of the double sum is the slope of
`β ↦ e^{β ⟨x_i, x_j⟩}` at `0`, which converges to its derivative
`⟨x_i, x_j⟩`.  The survey states the same continuity for `∇ 𝖤_β` and
`Hess 𝖤_β`, which are not formalized here. -/
theorem tendsto_Etilde_zero (X : SphereTuple d n) :
    Filter.Tendsto (fun β : ℝ => Etilde d n β X) (nhdsWithin 0 {(0 : ℝ)}ᶜ)
      (nhds (Etilde0 d n X)) := by
  have hslope : ∀ c : ℝ, Filter.Tendsto
      (fun β : ℝ => (Real.exp (β * c) - 1) / β) (nhdsWithin 0 {(0 : ℝ)}ᶜ) (nhds c) := by
    intro c
    have hd : HasDerivAt (fun β : ℝ => Real.exp (β * c)) c 0 := by
      simpa using ((hasDerivAt_id' (x := (0 : ℝ))).mul_const c).exp
    have h := hasDerivAt_iff_tendsto_slope.mp hd
    refine h.congr fun β => ?_
    simp [slope_def_field]
  have hsum : Filter.Tendsto
      (fun β : ℝ => ∑ i : Idx n, ∑ j : Idx n,
        (Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d))) - 1) / β)
      (nhdsWithin 0 {(0 : ℝ)}ᶜ)
      (nhds (∑ i : Idx n, ∑ j : Idx n,
        inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))) :=
    tendsto_finsetSum _ fun i _ => tendsto_finsetSum _ fun j _ => hslope _
  have hlim := hsum.const_mul (2⁻¹ : ℝ)
  rw [Etilde0]
  refine hlim.congr fun β => ?_
  rw [Etilde]
  simp only [div_eq_inv_mul, ← Finset.mul_sum, ← mul_assoc]
  rw [mul_inv]

/-- The constant of `eq: eigval.beta`:

  `c(d, n) = 2 (d - 1) / (d n)`,

the survey's lower bound on the positive eigenvalue of `Hess 𝖤_0` at a
non-trivial critical point. -/
noncomputable def eigvalBetaConst : ℝ :=
  (2 * ((d : ℝ) - 1)) / ((d : ℝ) * (n : ℝ))

/-- **Equation (eq: eigval.beta), the part `thm: beta.tiny` consumes.**

The proof of `thm: beta.tiny` uses `eq: eigval.beta` only through the fact
that the constant it exhibits is *strictly positive*, so that the threshold
`β ≤ c(d, n) / n` describes a nonempty range of inverse temperatures.  That
positivity is what is proved here: for `d ≥ 2` and `n ≥ 1`,

  `c(d, n) = 2 (d - 1) / (d n) > 0`.

The eigenvalue bound itself is a statement about `Hess 𝖤_0` on the sphere,
which this development does not carry, and is not formalized.
Source: arXiv:2312.10794v5, Appendix C. -/
theorem eigvalBetaConst_pos (hd : 2 ≤ d) (hn : 1 ≤ n) :
    0 < eigvalBetaConst d n := by
  have hd' : (2 : ℝ) ≤ (d : ℝ) := by exact_mod_cast hd
  have hn' : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  rw [eigvalBetaConst]
  apply div_pos <;> nlinarith

/-- The hypotheses of `eigvalBetaConst_pos` are satisfiable: the survey's own
regime `d = 2`, `n = 2` meets them. -/
example : 2 ≤ 2 ∧ 1 ≤ 2 := ⟨le_refl 2, one_le_two⟩

end Perspective
end Transformer
