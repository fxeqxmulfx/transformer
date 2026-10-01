/-
# Rescaling normalized attention dynamics by a change of time

Proposition `sprop: time_change` from Appendix D of arXiv:2510.22026v2
is the chain rule. The convergence proof is developed separately in
`Lojasiewicz` and its supporting modules.
-/

import Transformer.Normalization.Basic
import Mathlib.Analysis.Calculus.Deriv.Comp

namespace Transformer.Normalization

variable (d n : ℕ)

/-- **Proposition (sprop: time_change).** *A monotone time change rescales the
speed factors.*

For a strictly increasing `t(τ)` with positive derivative, the normalized
attention dynamics with speed regulation factors `s_j(t)` read in the time `τ`
is the normalized attention dynamics with speed regulation factors

  `s̃_j(τ) = s_j(t(τ)) / t'(τ)`,

the time-dependent parameters being reparametrized along with it.

Only differentiability of `t(τ)` enters the identity: it is the chain rule,
`θ̇(t(τ)) = t'(τ) · θ̇|_{t(τ)}`, and the division by `t'(τ)` in `s̃_j` is what
absorbs that factor — at `t'(τ) = 0` both sides read `0`. Strict monotonicity,
positivity of `t'` and growth of `t(τ)` to infinity are what make the change of
variable a reparametrization of the whole half-line, which is the use the paper
makes of it; none of them is needed here, so none is among the hypotheses.

Source: arXiv:2510.22026v2, Appendix D, `sprop: time_change`. -/
theorem na_time_change
    (β : ℝ) (Q K V : ℝ → ParamMatrix d) (s : ℝ → Idx n → ℝ)
    (θ : ℝ → Idx n → EucSpace d) (tOf tOf' : ℝ → ℝ)
    (hderiv : ∀ u : ℝ, HasDerivAt tOf (tOf' u) u)
    (hNA : NA d n β Q K V s θ) :
    NA d n β (fun u => Q (tOf u)) (fun u => K (tOf u)) (fun u => V (tOf u))
      (fun u j => s (tOf u) j / tOf' u) (fun u => θ (tOf u)) := by
  intro u j
  have h := (hNA (tOf u) j).scomp u (hderiv u)
  rw [show (s (tOf u) j / tOf' u)⁻¹ = tOf' u * (s (tOf u) j)⁻¹ by
      rw [inv_div, div_eq_mul_inv], mul_smul]
  exact h

/-- All hypotheses of `na_time_change` are satisfiable: the identity time
change has derivative `1` everywhere, and the frozen dynamics `s ≡ 0`, `θ ≡ 0`
solves `eq: NA` — at `s_j = 0` the prescribed velocity is `0⁻¹ • _ = 0`, which
is the derivative of a constant. Source: arXiv:2510.22026v2, Appendix D,
`sprop: time_change`. -/
example :
    (∀ u : ℝ, HasDerivAt (id : ℝ → ℝ) 1 u) ∧
      NA 1 1 0 (fun _ => ContinuousLinearMap.id ℝ (EucSpace 1))
        (fun _ => ContinuousLinearMap.id ℝ (EucSpace 1))
        (fun _ => ContinuousLinearMap.id ℝ (EucSpace 1))
        (fun _ _ => 0) (fun _ _ => 0) := by
  refine ⟨fun u => hasDerivAt_id u, fun t j => ?_⟩
  simpa using hasDerivAt_const t (0 : EucSpace 1)

end Transformer.Normalization
