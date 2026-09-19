/-
# Appendix D — the assembly of the phase-transition curve

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

One statement: `e:ineqsecondpart`, the second half of `eq: upto-t`.  It is
proved — as the deduction it is, from `e:productcloseto1` and
`e:ybetacloseto1` carried as explicit hypotheses, and with the constant its own
derivation supports rather than the one the survey prints.

The differential inequality `e:diffineqalpha` and its integrated form
`e:productcloseto1` are in `Perspective.AppendixD_Product`; the `USA` analogue
`rem: usa.d` is in `Perspective.AppendixD_YbetaUSA`.

One range to watch: `hcp` below is asked on `t ≥ 0`, which is where the survey
puts `e:productcloseto1`, while `Perspective.product_close_to_one` proves that
estimate only on `t ≥ 1/n` — on `[0, 1/n)` it is false, and
`Perspective.not_forall_product_close_to_one` refutes it there.  The deduction
below is stated at its full strength; the chain it belongs to runs on
`t ≥ 1/n`.

The assembly of `thm: phase.transition.curve`
out of `e:ineqfirstpart` and `e:ineqsecondpart` is not a separate statement: the
theorem it assembles is `Perspective.phase_transition_curve`, itself sorried, so
an implication into it would be provable in one line and would assert nothing.

The witnesses below are all at `d = 1`, `n = 2`, `β = 0`, where the consensus
configuration solves `SA` and `tanh` solves `eq: ybeta`
(`Perspective.ybetaODE_SA_two_zero`).
-/

import Transformer.Perspective.AppendixD_Product

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **Equation (e:ineqsecondpart).** *Second half of `eq: upto-t`.*

  `|⟨x_i(t), x_j(t)⟩ - γ_β(t)|
      ≤ 4 exp((1 - γ_β(1/n) t) / (2 n e^{2β}))
        + (1/2) exp( n² e^β / (2(n + e^{β/2})) - n t / (n + e^{β/2}) )`.

**What the source says and what is changed here.**  Two changes, both forced
by the survey's own derivation.

*The two inputs are carried as hypotheses.*  The survey deduces this from
`e:productcloseto1` (`product_close_to_one`) and `e:ybetacloseto1`
(`ybeta_close_to_1`).  They enter as `hcp` and `hyb`, so that what is proved is
the deduction and its dependence is visible in the signature.  `hγle`, that `γ_β ≤ 1`, is the remaining property of the
solution of `eq: ybeta` the step uses, and is carried the same way.

*The constant on the first summand is `4`, not `1`.*  `α` bounds the inner
products with `x⋆`, not the pairwise ones, and the passage costs a factor:
from `1 - ⟨x_i, x⋆⟩ ≤ ε` and `1 - ⟨x_j, x⋆⟩ ≤ ε` one gets
`‖x_i - x⋆‖, ‖x_j - x⋆‖ ≤ √(2ε)`, hence `‖x_i - x_j‖ ≤ 2√(2ε)` and
`1 - ⟨x_i, x_j⟩ = ‖x_i - x_j‖²/2 ≤ 4ε`.  The survey writes the conclusion with
`1` in place of `4`; nothing in its argument supplies that, and the `C e^{-λt}`
branch of `eq: upto-t` the estimate feeds is unaffected by the constant.

The restriction to `i ≠ j` is dropped: the bound holds for every pair, the
diagonal included.

Source: arXiv:2312.10794v5, Appendix D, `e:ineqsecondpart`. -/
theorem ineq_second_part (β : ℝ) (X : ℝ → SphereTuple d n) (γ α : ℝ → ℝ)
    (x_star : SSphere d) (hα : IsMinInner d n X x_star α)
    (hγle : ∀ t : ℝ, 0 ≤ t → γ t ≤ 1)
    (hcp : ∀ t : ℝ, 0 ≤ t →
      1 - α t
        ≤ Real.exp ((1 - γ ((n : ℝ)⁻¹) * t) / (2 * (n : ℝ) * Real.exp (2 * β))))
    (hyb : ∀ t : ℝ, 0 ≤ t →
      1 - γ t
        ≤ (1/2 : ℝ) * Real.exp
            ((n : ℝ)^2 * Real.exp β / (2 * ((n : ℝ) + Real.exp (β / 2)))
              - (n : ℝ) * t / ((n : ℝ) + Real.exp (β / 2)))) :
    ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n,
      |inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d)) - γ t|
        ≤ 4 * Real.exp ((1 - γ ((n : ℝ)⁻¹) * t) / (2 * (n : ℝ) * Real.exp (2 * β)))
          + (1/2 : ℝ) * Real.exp
              ((n : ℝ)^2 * Real.exp β / (2 * ((n : ℝ) + Real.exp (β / 2)))
                - (n : ℝ) * t / ((n : ℝ) + Real.exp (β / 2))) := by
  intro t ht i j
  have hx : ‖((X t i : SSphere d) : EucSpace d)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (X t i).2
  have hy : ‖((X t j : SSphere d) : EucSpace d)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (X t j).2
  have hw : ‖((x_star : SSphere d) : EucSpace d)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp x_star.2
  -- Cauchy--Schwarz on unit vectors
  have hxy1 : inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d)) ≤ 1 := by
    have := real_inner_le_norm ((X t i : EucSpace d)) ((X t j : EucSpace d))
    rwa [hx, hy, one_mul] at this
  have hxw1 : inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((x_star : EucSpace d)) ≤ 1 := by
    have := real_inner_le_norm ((X t i : EucSpace d)) ((x_star : EucSpace d))
    rwa [hx, hw, one_mul] at this
  -- `α t` bounds both inner products with `x⋆`
  have hax : α t ≤ inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((x_star : EucSpace d)) :=
    (hα t).1 i
  have hay : α t ≤ inner (𝕜 := ℝ) ((X t j : EucSpace d)) ((x_star : EucSpace d)) :=
    (hα t).1 j
  -- the two sides of the triangle, squared
  have hA : ‖((X t i : EucSpace d)) - ((x_star : EucSpace d))‖ ^ 2 ≤ 2 * (1 - α t) := by
    rw [norm_sub_sq_real, hx, hw]
    linarith
  have hB : ‖((x_star : EucSpace d)) - ((X t j : EucSpace d))‖ ^ 2 ≤ 2 * (1 - α t) := by
    rw [norm_sub_sq_real, hw, hy,
      real_inner_comm ((X t j : EucSpace d)) ((x_star : EucSpace d))]
    linarith
  have htri : ‖((X t i : EucSpace d)) - ((X t j : EucSpace d))‖
      ≤ ‖((X t i : EucSpace d)) - ((x_star : EucSpace d))‖
        + ‖((x_star : EucSpace d)) - ((X t j : EucSpace d))‖ := by
    calc ‖((X t i : EucSpace d)) - ((X t j : EucSpace d))‖
        = ‖(((X t i : EucSpace d)) - ((x_star : EucSpace d)))
            + (((x_star : EucSpace d)) - ((X t j : EucSpace d)))‖ := by
          rw [sub_add_sub_cancel]
      _ ≤ _ := norm_add_le _ _
  have hsq : ‖((X t i : EucSpace d)) - ((X t j : EucSpace d))‖ ^ 2
      ≤ (‖((X t i : EucSpace d)) - ((x_star : EucSpace d))‖
          + ‖((x_star : EucSpace d)) - ((X t j : EucSpace d))‖) ^ 2 := by
    nlinarith [norm_nonneg (((X t i : EucSpace d)) - ((X t j : EucSpace d))),
      norm_nonneg (((X t i : EucSpace d)) - ((x_star : EucSpace d))),
      norm_nonneg (((x_star : EucSpace d)) - ((X t j : EucSpace d))), htri]
  have hexp : ‖((X t i : EucSpace d)) - ((X t j : EucSpace d))‖ ^ 2
      = 2 - 2 * inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d)) := by
    rw [norm_sub_sq_real, hx, hy]
    ring
  -- the factor 4
  have key : 1 - inner (𝕜 := ℝ) ((X t i : EucSpace d)) ((X t j : EucSpace d))
      ≤ 4 * (1 - α t) := by
    nlinarith [hsq, hexp, hA, hB,
      sq_nonneg (‖((X t i : EucSpace d)) - ((x_star : EucSpace d))‖
        - ‖((x_star : EucSpace d)) - ((X t j : EucSpace d))‖)]
  have hE1 : (0 : ℝ)
      ≤ Real.exp ((1 - γ ((n : ℝ)⁻¹) * t) / (2 * (n : ℝ) * Real.exp (2 * β))) :=
    (Real.exp_pos _).le
  have hE2 : (0 : ℝ) ≤ (1/2 : ℝ) * Real.exp
      ((n : ℝ)^2 * Real.exp β / (2 * ((n : ℝ) + Real.exp (β / 2)))
        - (n : ℝ) * t / ((n : ℝ) + Real.exp (β / 2))) := by positivity
  rw [abs_le]
  exact ⟨by linarith [hcp t ht, hγle t ht], by linarith [hyb t ht]⟩

/-- The hypotheses of `ineq_second_part` are satisfiable: the two particles of
`isMinInner_const_consensus`, where `α ≡ 1`, together with `γ ≡ 1`, which
meets `hγle`, `hcp` and `hyb` because both left-hand sides are `0`. -/
example :
    IsMinInner 1 2 (fun _ _ => basePoint 0) (basePoint 0) (fun _ => 1) ∧
      (∀ t : ℝ, 0 ≤ t → (fun _ : ℝ => (1 : ℝ)) t ≤ 1) ∧
      (∀ t : ℝ, 0 ≤ t →
        1 - (fun _ : ℝ => (1 : ℝ)) t
          ≤ Real.exp ((1 - (fun _ : ℝ => (1 : ℝ)) (((2 : ℕ) : ℝ)⁻¹) * t)
              / (2 * ((2 : ℕ) : ℝ) * Real.exp (2 * (0 : ℝ))))) ∧
      (∀ t : ℝ, 0 ≤ t →
        1 - (fun _ : ℝ => (1 : ℝ)) t
          ≤ (1/2 : ℝ) * Real.exp
              (((2 : ℕ) : ℝ)^2 * Real.exp 0
                  / (2 * (((2 : ℕ) : ℝ) + Real.exp ((0 : ℝ) / 2)))
                - ((2 : ℕ) : ℝ) * t / (((2 : ℕ) : ℝ) + Real.exp ((0 : ℝ) / 2)))) := by
  refine ⟨isMinInner_const_consensus 2 two_pos, fun _ _ => le_rfl, fun _ _ => ?_,
    fun _ _ => ?_⟩ <;>
  · simp only [sub_self]
    positivity

end Perspective
end Transformer
