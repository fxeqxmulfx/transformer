/-
# Appendix D — the smallest coordinate `α(t)`, and the assembly

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

The second half of Appendix D of the survey:

* `e:1/n`            — the lower bound `α(1/n) ≥ γ_β(1/n)/2`,
* `e:dotalpha`, `e:mineqalpha`, `e:diffineqalpha`
                     — the differential inequality satisfied by `α`,
* `e:productcloseto1` — its integrated form,
* `e:ineqsecondpart`  — the second half of `eq: upto-t`,
* the assembly of `thm: phase.transition.curve`,
* `rem: usa.d`        — the analogue for `USA`.

The stability estimates these rest on are in
`Perspective.AppendixD_PhaseTransition`.
-/

import Transformer.Perspective.AppendixD_PhaseTransition

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **Equation (e:1/n).**

  `α(1/n) ≥ (1/2) γ_β(1/n)`.

At time `1/n` the configuration is already within half of the orthogonal
angle `γ_β(1/n)` of the direction `x⋆`, for a suitable `x⋆`.

Not proved here: it is `e:ineqfirstpart` at `t = 1/n` combined with
`eq: d.large`, which makes the error `2 c(β)^{1} √(log d / d)` smaller than
`γ_β(1/n)/2`.

The survey states this for `n ≥ 2`; no such restriction is needed for the
inequality itself, and dropping it is what makes the hypotheses satisfiable at
the one solution of `eq: ybeta` available in closed form.

Source: arXiv:2312.10794v5, Appendix D, `e:1/n`. -/
theorem alpha_at_one_over_n
    (β : ℝ) (γ α : ℝ → ℝ) (X : ℝ → SphereTuple d n) (x_star : SSphere d)
    (hX : SA d n β X) (hγ : ybetaODE_SA n β γ)
    (hα : IsMinInner d n X x_star α) :
    (1/2 : ℝ) * γ ((n : ℝ)⁻¹) ≤ α ((n : ℝ)⁻¹) := by
  sorry

/-- The hypotheses of `alpha_at_one_over_n` are satisfiable: the consensus
solution at `d = n = 1`, the closed-form solution of `eq: ybeta`, and the
constant `α ≡ 1 = ⟨x, x⟩`. -/
example :
    SA 1 1 0 (fun _ _ => basePoint 0) ∧
      ybetaODE_SA 1 0 (fun t => 1 - Real.exp (-2 * t)) ∧
      IsMinInner 1 1 (fun _ _ => basePoint 0) (basePoint 0) (fun _ => 1) := by
  refine ⟨SA_const_consensus 1 1 one_pos 0 (basePoint 0), ybetaODE_SA_one_zero,
    fun t => ?_⟩
  have hx : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  have hxx : inner (𝕜 := ℝ) (((basePoint 0 : SSphere 1)) : EucSpace 1)
      (((basePoint 0 : SSphere 1)) : EucSpace 1) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  exact ⟨fun _ => le_of_eq hxx.symm, ⟨0, hxx.symm⟩⟩

/-- **Equation (e:diffineqalpha).** *The differential inequality for `α`.*

  `α̇(t) ≥ (1/(n e^{2β})) α(1/n) (1 - α(t))`   for `t ≥ 1/n`.

A `Prop`-valued definition and not a theorem: `α` is a minimum of finitely many
smooth functions, so the survey argues with its lower Dini derivative; the
statement below asserts, in addition, that `α` is differentiable, and neither
half is proved here.

Source: arXiv:2312.10794v5, Appendix D, `e:mineqalpha`, `e:diffineqalpha`. -/
def DiffIneqAlpha (β : ℝ) : Prop :=
  ∀ (X : ℝ → SphereTuple d n) (x_star : SSphere d) (α : ℝ → ℝ),
    SA d n β X → IsMinInner d n X x_star α →
    ∀ t : ℝ, (n : ℝ)⁻¹ ≤ t →
      ∃ c : ℝ, HasDerivAt α c t ∧
        ((n : ℝ) * Real.exp (2 * β))⁻¹ * α ((n : ℝ)⁻¹) * (1 - α t) ≤ c

/-- **Equation (e:productcloseto1).**

  `1 - α(t) ≤ exp( (1 - γ_β(1/n) t) / (2 n e^{2β}) )`.

A `Prop`-valued definition and not a theorem: it is `e:diffineqalpha`
integrated by Grönwall, starting from `e:1/n`, neither of which is proved here.

Source: arXiv:2312.10794v5, Appendix D, `e:productcloseto1`. -/
def ProductCloseToOne (β : ℝ) : Prop :=
  ∀ (X : ℝ → SphereTuple d n) (γ α : ℝ → ℝ) (x_star : SSphere d),
    SA d n β X → ybetaODE_SA n β γ → IsMinInner d n X x_star α →
    ∀ t : ℝ, 0 ≤ t →
      1 - α t
        ≤ Real.exp ((1 - γ ((n : ℝ)⁻¹) * t) / (2 * (n : ℝ) * Real.exp (2 * β)))

/-- **Equation (e:ineqsecondpart).** *Second half of `eq: upto-t`.*

  `|⟨x_i(t), x_j(t)⟩ - γ_β(t)|
      ≤ exp((1 - γ_β(1/n) t) / (2 n e^{2β}))
        + (1/2) exp( n² e^β / (2(n + e^{β/2})) - n t / (n + e^{β/2}) )`.

The first summand bounds `|⟨x_i, x_j⟩ - 1|` through `e:productcloseto1`, the
second `|1 - γ_β(t)|` through `e:ybetacloseto1`; both terms decay
exponentially in `t`, which is the `C e^{-λt}` branch of `eq: upto-t`.

A `Prop`-valued definition and not a theorem: it rests on
`ProductCloseToOne`, which is not proved here.

Source: arXiv:2312.10794v5, Appendix D, `e:ineqsecondpart`. -/
def IneqSecondPart (β : ℝ) : Prop :=
  ∀ (X : ℝ → SphereTuple d n) (γ α : ℝ → ℝ) (x_star : SSphere d),
    SA d n β X → ybetaODE_SA n β γ → IsMinInner d n X x_star α →
    ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n, i ≠ j →
      |inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d)) - γ t|
        ≤ Real.exp ((1 - γ ((n : ℝ)⁻¹) * t) / (2 * (n : ℝ) * Real.exp (2 * β)))
          + (1/2 : ℝ) * Real.exp
              ((n : ℝ)^2 * Real.exp β / (2 * ((n : ℝ) + Real.exp (β / 2)))
                - (n : ℝ) * t / ((n : ℝ) + Real.exp (β / 2)))

/-- *Assembly of `thm: phase.transition.curve`.*  Past the threshold
`eq: d.large`, `e:ineqfirstpart` gives the `√(log d / d)` branch of `eq: upto-t`
and `e:ineqsecondpart` the exponentially decaying branch; the theorem is their
minimum.

A `Prop`-valued definition and not a theorem: the implication is stated, not
proved, and neither is `IneqSecondPart` which it consumes.

Source: arXiv:2312.10794v5, Appendix D. -/
def PhaseTransitionProofAssembly : Prop :=
  2 ≤ n → ∀ β : ℝ, 0 ≤ β → ∀ γ : ℝ → ℝ, ybetaODE_SA n β γ →
    16 * (cBeta β)^2 / (γ ((n : ℝ)⁻¹))^2 ≤ (d : ℝ) / Real.log d →
    AlmostOrthogonal d n → IneqSecondPart d n β →
      ∃ C lam : ℝ, 0 < C ∧ 0 < lam ∧
        ∀ (X : ℝ → SphereTuple d n) (α : ℝ → ℝ) (x_star : SSphere d),
          SA d n β X → IsMinInner d n X x_star α →
          ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n, i ≠ j →
            |inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d)) - γ t|
              ≤ min (2 * cBeta β ^ ((n : ℝ) * t) * Real.sqrt (Real.log d / d))
                    (C * Real.exp (-(lam * t)))

/-- **Remark (rem: usa.d).** *The analogue for `USA`.*

The same argument runs with `eq: ybeta` replaced by `eq: ybetaUSA`; there the
angle closes at the cleaner rate

  `1 - γ_β(t) ≤ (1/2) exp(-e^{β/2} (t - n/2))`.

A `Prop`-valued definition and not a theorem: the `USA` estimate is not proved
here.

Source: arXiv:2312.10794v5, Appendix D, `rem: usa.d`. -/
def UsaAnalogue : Prop :=
  ∀ (β : ℝ) (γ : ℝ → ℝ), 0 ≤ β → ybetaODE_USA n β γ →
    ∀ t : ℝ, 0 ≤ t →
      1 - γ t ≤ (1/2 : ℝ) * Real.exp (-(Real.exp (β / 2) * (t - (n : ℝ) / 2)))

end Perspective
end Transformer
