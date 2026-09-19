/-
# Open Problem `o:strictsaddle` — the critical points of `𝖤_β` on `𝕋^n`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

§7 asks whether, apart from the global maxima, every critical point of the
toroidal energy `𝖤_β` is a strict saddle — a point at which some direction
sees a strictly positive second derivative, so that gradient ascent escapes.

The survey leaves it open.  Open is a kind of unproved, and the sorry count
is where unproved is recorded, so it is stated here as a theorem with
`sorry`: nothing in this development may be built on it.

The vocabulary is Appendix B's: `Perspective.SecondDerivTorusEnergy` is the
second derivative along a straight line of `𝕋^n`, and
`Perspective.TorusHessianNonPos` its non-positivity, which is what a critical
point that is *not* a strict saddle satisfies.
-/

import Transformer.Perspective.AppendixB_Taylor

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (n : ℕ)

/-- *A critical point of `𝖤_β` on `𝕋^n`*: every partial derivative vanishes.

The partial derivative is the one `hasDerivAt_torusEnergy` computes, so this
says `Σ_j e^{β cos(θ_j - θ_i)} sin(θ_j - θ_i) = 0` for every `i`, written as
a `HasDerivAt` so that no `deriv` of a non-differentiable function can slip
in.

Source: arXiv:2312.10794v5, §7, `o:strictsaddle`. -/
def IsCriticalTorusEnergy (β : ℝ) (θ : Angles n) : Prop :=
  ∀ i : Idx n,
    HasDerivAt (fun x : ℝ => torusEnergy n β (Function.update θ i x)) 0 (θ i)

/-- *A global maximum of `𝖤_β` on `𝕋^n`.*  The dynamics `eq:onangles` is a
gradient **ascent** (`angularUSA_is_gradient_flow`), so the maxima are the
attractors and the open problem is about everything else. -/
def IsGlobalMaxTorusEnergy (β : ℝ) (θ : Angles n) : Prop :=
  ∀ φ : Angles n, torusEnergy n β φ ≤ torusEnergy n β θ

/-- *A strict saddle of `𝖤_β`*: some straight line through `θ` sees a
strictly positive second derivative, so the ascent leaves along it.

This is the negation of `TorusHessianNonPos` sharpened from `¬ (c ≤ 0)` to an
exhibited direction, which is what "strict" asks for. -/
def IsStrictSaddleTorusEnergy (β : ℝ) (θ : Angles n) : Prop :=
  ∃ (v : Angles n) (c : ℝ), SecondDerivTorusEnergy n β θ v c ∧ 0 < c

/-- A strict saddle has no non-positive Hessian, which is the form Appendix B
uses its hypothesis in: `taylor2_inequality` and `taylor3_inequality` say
nothing at a strict saddle. -/
theorem not_torusHessianNonPos_of_strictSaddle (β : ℝ) (θ : Angles n)
    (h : IsStrictSaddleTorusEnergy n β θ) : ¬ TorusHessianNonPos n β θ := by
  obtain ⟨v, c, hc, hcpos⟩ := h
  exact fun hhess => absurd (hhess v c hc) (not_le.mpr hcpos)

/-- **Open Problem (o:strictsaddle).** *Non-maximal critical points are
strict saddles.*

For `n ≥ 2` and `β > 0`, is every critical point of `𝖤_β` on `𝕋^n` that is
not a global maximum a strict saddle?

An affirmative answer would make the gradient ascent `eq:onangles` converge
to a global maximum from almost every initial configuration, which is the
`d = 2` clustering the survey is after.

The two-particle antipodal configuration is the picture: at `n = 2`,
`θ = (0, π)` the energy is critical and minimal, and rotating one particle
raises it at rate `β e^{-β} > 0`.

Not proved here; the survey leaves it open.

Source: arXiv:2312.10794v5, §7, `o:strictsaddle`. -/
theorem strict_saddle (β : ℝ) (hβ : 0 < β) (hn : 2 ≤ n) :
    ∀ θ : Angles n, IsCriticalTorusEnergy n β θ →
      ¬ IsGlobalMaxTorusEnergy n β θ → IsStrictSaddleTorusEnergy n β θ := by
  sorry

/-- The hypotheses of `strict_saddle` are satisfiable: `β = 1`, `n = 2`. -/
example : (0 : ℝ) < 1 ∧ 2 ≤ 2 := ⟨one_pos, le_rfl⟩

end Perspective
end Transformer
