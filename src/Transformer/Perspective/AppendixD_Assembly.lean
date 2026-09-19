/-
# Appendix D — the assembly of the phase-transition curve

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

What the survey does with the estimate `e:1/n` of `Perspective.AppendixD_Alpha`:

* `e:mineqalpha`, `e:diffineqalpha` — the differential inequality for `α`,
* `e:productcloseto1`              — its integrated form,
* `e:ineqsecondpart`               — the second half of `eq: upto-t`,
* `rem: usa.d`                     — the analogue for `USA`.

None of the four is proved here.  The assembly of `thm: phase.transition.curve`
out of `e:ineqfirstpart` and `e:ineqsecondpart` is not a separate statement: the
theorem it assembles is `Perspective.phase_transition_curve`, itself sorried, so
an implication into it would be provable in one line and would assert nothing.

The witnesses below are all at `d = 1`, `n = 2`, `β = 0`, where the consensus
configuration solves `SA` and `tanh` solves `eq: ybeta`
(`Perspective.ybetaODE_SA_two_zero`); two particles is what makes the `i ≠ j`
of `ineq_second_part` satisfiable.
-/

import Transformer.Perspective.AppendixD_Alpha

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- The consensus configuration of `n` copies of `basePoint 0` in `𝕊^0`, with
its minimum-inner-product function `α ≡ 1`: the common witness of the `SA` and
`IsMinInner` hypotheses of this file. -/
theorem isMinInner_const_consensus (m : ℕ) (hm : 0 < m) :
    IsMinInner 1 m (fun _ _ => basePoint 0) (basePoint 0) (fun _ => 1) := by
  have hx : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  have hxx : inner (𝕜 := ℝ) (((basePoint 0 : SSphere 1)) : EucSpace 1)
      (((basePoint 0 : SSphere 1)) : EucSpace 1) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  exact fun _ => ⟨fun _ => le_of_eq hxx.symm, ⟨⟨0, hm⟩, hxx.symm⟩⟩

/-- **Equation (e:diffineqalpha).** *The differential inequality for `α`.*

  `α̇(t) ≥ (1/(n e^{2β})) α(1/n) (1 - α(t))`   for `t ≥ 1/n`.

`α` is a minimum of finitely many smooth functions, so the survey argues with
its lower Dini derivative; the statement below asserts, in addition, that `α`
is differentiable.

Not proved here.

Source: arXiv:2312.10794v5, Appendix D, `e:mineqalpha`, `e:diffineqalpha`. -/
theorem diff_ineq_alpha (β : ℝ) (X : ℝ → SphereTuple d n) (x_star : SSphere d)
    (α : ℝ → ℝ) (hX : SA d n β X) (hα : IsMinInner d n X x_star α) :
    ∀ t : ℝ, (n : ℝ)⁻¹ ≤ t →
      ∃ c : ℝ, HasDerivAt α c t ∧
        ((n : ℝ) * Real.exp (2 * β))⁻¹ * α ((n : ℝ)⁻¹) * (1 - α t) ≤ c := by
  sorry

/-- The hypotheses of `diff_ineq_alpha` are satisfiable: two particles sitting
together at `basePoint 0`, where `α ≡ 1`. -/
example :
    SA 1 2 0 (fun _ _ => basePoint 0) ∧
      IsMinInner 1 2 (fun _ _ => basePoint 0) (basePoint 0) (fun _ => 1) :=
  ⟨SA_const_consensus 1 2 two_pos 0 (basePoint 0), isMinInner_const_consensus 2 two_pos⟩

/-- **Equation (e:productcloseto1).**

  `1 - α(t) ≤ exp( (1 - γ_β(1/n) t) / (2 n e^{2β}) )`.

This is `e:diffineqalpha` integrated by Grönwall, starting from `e:1/n`.

Not proved here.

Source: arXiv:2312.10794v5, Appendix D, `e:productcloseto1`. -/
theorem product_close_to_one (β : ℝ) (X : ℝ → SphereTuple d n) (γ α : ℝ → ℝ)
    (x_star : SSphere d) (hX : SA d n β X) (hγ : ybetaODE_SA n β γ)
    (hα : IsMinInner d n X x_star α) :
    ∀ t : ℝ, 0 ≤ t →
      1 - α t
        ≤ Real.exp ((1 - γ ((n : ℝ)⁻¹) * t) / (2 * (n : ℝ) * Real.exp (2 * β))) := by
  sorry

/-- The hypotheses of `product_close_to_one` are satisfiable: the same two
particles, and `γ = tanh`, which solves `eq: ybeta` at `n = 2`, `β = 0`. -/
example :
    SA 1 2 0 (fun _ _ => basePoint 0) ∧ ybetaODE_SA 2 0 Real.tanh ∧
      IsMinInner 1 2 (fun _ _ => basePoint 0) (basePoint 0) (fun _ => 1) :=
  ⟨SA_const_consensus 1 2 two_pos 0 (basePoint 0), ybetaODE_SA_two_zero,
    isMinInner_const_consensus 2 two_pos⟩

/-- **Equation (e:ineqsecondpart).** *Second half of `eq: upto-t`.*

  `|⟨x_i(t), x_j(t)⟩ - γ_β(t)|
      ≤ exp((1 - γ_β(1/n) t) / (2 n e^{2β}))
        + (1/2) exp( n² e^β / (2(n + e^{β/2})) - n t / (n + e^{β/2}) )`.

The first summand bounds `|⟨x_i, x_j⟩ - 1|` through `e:productcloseto1`, the
second `|1 - γ_β(t)|` through `e:ybetacloseto1`; both terms decay
exponentially in `t`, which is the `C e^{-λt}` branch of `eq: upto-t`.

Not proved here; it rests on `product_close_to_one`, which is not proved
either.

Source: arXiv:2312.10794v5, Appendix D, `e:ineqsecondpart`. -/
theorem ineq_second_part (β : ℝ) (X : ℝ → SphereTuple d n) (γ α : ℝ → ℝ)
    (x_star : SSphere d) (hX : SA d n β X) (hγ : ybetaODE_SA n β γ)
    (hα : IsMinInner d n X x_star α) :
    ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n, i ≠ j →
      |inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d)) - γ t|
        ≤ Real.exp ((1 - γ ((n : ℝ)⁻¹) * t) / (2 * (n : ℝ) * Real.exp (2 * β)))
          + (1/2 : ℝ) * Real.exp
              ((n : ℝ)^2 * Real.exp β / (2 * ((n : ℝ) + Real.exp (β / 2)))
                - (n : ℝ) * t / ((n : ℝ) + Real.exp (β / 2))) := by
  sorry

/-- The hypotheses of `ineq_second_part` are satisfiable, and its conclusion is
not vacuous at the witness: with two particles the pair `i ≠ j` exists. -/
example :
    SA 1 2 0 (fun _ _ => basePoint 0) ∧ ybetaODE_SA 2 0 Real.tanh ∧
      IsMinInner 1 2 (fun _ _ => basePoint 0) (basePoint 0) (fun _ => 1) ∧
      ∃ i j : Idx 2, i ≠ j :=
  ⟨SA_const_consensus 1 2 two_pos 0 (basePoint 0), ybetaODE_SA_two_zero,
    isMinInner_const_consensus 2 two_pos, 0, 1, by decide⟩

/-- **Remark (rem: usa.d).** *The analogue for `USA`.*

The same argument runs with `eq: ybeta` replaced by `eq: ybetaUSA`; there the
angle closes at the cleaner rate

  `1 - γ_β(t) ≤ (1/2) exp(-e^{β/2} (t - n/2))`.

The hypothesis `2 ≤ n` is not decoration: at `n = 1`, `β = 0`, `t = 0` the
bound reads `1 - γ(0) = 1 ≤ (1/2) e^{1/2} ≈ 0.824`, which is false.  The
remark is about the `n ≥ 2` regime of `thm: phase.transition.curve`.

Not proved here.

Source: arXiv:2312.10794v5, Appendix D, `rem: usa.d`. -/
theorem usa_analogue (hn : 2 ≤ n) (β : ℝ) (γ : ℝ → ℝ) (hβ : 0 ≤ β)
    (hγ : ybetaODE_USA n β γ) :
    ∀ t : ℝ, 0 ≤ t →
      1 - γ t ≤ (1/2 : ℝ) * Real.exp (-(Real.exp (β / 2) * (t - (n : ℝ) / 2))) := by
  sorry

/-- The hypotheses of `usa_analogue` are satisfiable at the smallest `n` it
allows: `n = 2`, `β = 0`, `γ = tanh`, where at `t = 0` the conclusion reads
`1 ≤ e/2 ≈ 1.359`. -/
example : 2 ≤ 2 ∧ (0 : ℝ) ≤ 0 ∧ ybetaODE_USA 2 0 Real.tanh :=
  ⟨le_rfl, le_rfl, ybetaODE_USA_two_zero⟩

end Perspective
end Transformer
