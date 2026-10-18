/-
# Normalization — The convergence machinery of the proof (App. D of 2510.22026v2)

Two of the three statements the proof of `thm: convergence` rests on:

* `Lemma lem: loj` — the Łojasiewicz convergence theorem generalized to a
  modulated gradient flow `ẋ = -M(t) ∇E(x)`;
* `Proposition sprop: time_change` — a monotone time change turns `eq: NA`
  with speed factors `s_j` into `eq: NA` with `s_j(t(τ)) / t'(τ)`.

The second is the chain rule, proved in `TimeChange`. For the first, all dynamical
steps are proved in `LojasiewiczConditional`, given the local analytic gradient
inequality at every critical point. The inequality is immediate for locally constant
energies; otherwise local uniformization (`Transformer.Uniformization`) gives the energy
compact monomial charts (`AnalyticMonomialCharts`), from which `CompactMonomialCharts`
derives it. The modules on low dimensions, nondegenerate critical points, normal-crossing
charts and curve selection prove special cases and alternative reductions of the
inequality.
The third, `Lemma lem: matrix`, is proved in `Normalization.UnstableProduct`,
whose matrix square root neither of these two needs.
-/

import Transformer.Basic
import Transformer.Normalization.Basic
import Transformer.Normalization.LojasiewiczConditional
import Transformer.Normalization.ModulatedQuadratic
import Transformer.Normalization.ModulatedScalar
import Transformer.Normalization.NondegenerateGradientInequality
import Transformer.Normalization.BlowupGradientInequality
import Transformer.Normalization.MinimumCurveSelection
import Transformer.Normalization.AnalyticDirection
import Transformer.Normalization.AnalyticImplicitBranch
import Transformer.Normalization.EnergyLevelPreparation
import Transformer.Normalization.QuadraticCurveLifting
import Transformer.Normalization.AnalyticPlaneCurveSelection
import Transformer.Normalization.EnergyGradientMinimumLifting
import Transformer.Normalization.GlobalMinimumLiftingExamples
import Transformer.Normalization.AnalyticAtlasQueryExamples
import Transformer.Normalization.CompactBallGradientQueries
import Transformer.Normalization.AnalyticFiberValueRelation
import Transformer.Normalization.AnalyticMonomialCharts
import Transformer.RealAnalyticGerms
import Transformer.Sturm
import Transformer.Normalization.TimeChange
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace Normalization

/-- **Lemma (lem: loj).** *Łojasiewicz convergence for a modulated gradient
flow.*

Let `M(t)` be symmetric with `C λ(t) I ≻ M(t) ≻ λ(t) I`, `λ(t) > 0` and
`∫_0^∞ λ = ∞`, and let `E` be analytic on an open `U ⊆ ℝ^N`.  Then any
trajectory of

  `ẋ = -M(t) ∇_x E(x)`

that stays in a compact subset of `U` converges to a critical point of `E`.

The two matrix inequalities are written out on vectors, `λ(t) ‖v‖² < ⟨M(t) v, v⟩
< C λ(t) ‖v‖²` for `v ≠ 0`, rather than through a positive-definiteness order
on operators; the divergence of `∫ λ` is the divergence of its primitive.
The compact path is a compact `K ⊆ U` the trajectory never leaves.
Symmetry, like the matrix inequalities, is required only for `t ≥ 0`,
matching the manuscript's quantifier.

The proof takes from Łojasiewicz the local gradient inequality
`|E(y) - E(z)|^α ≤ k ‖∇E(y)‖` near each critical point `z`
(`analytic_gradient_inequality`). It is immediate for locally constant energies and
otherwise follows from compact monomial charts, which local uniformization provides
(`hasCompactMonomialCharts_of_analyticAt`). Everything after that inequality is proved by
`lojasiewicz_modulated_of_local_gradient_inequality`, including compact uniformization,
convergence of the trajectory, and criticality of the limit.

Source: arXiv:2510.22026v2, Appendix D.1, `lem: loj`. -/
theorem lojasiewicz_modulated
    (N : ℕ) (U : Set (EucSpace N)) (hU : IsOpen U)
    (E : EucSpace N → ℝ) (hE : AnalyticOnNhd ℝ E U)
    (C : ℝ) (lam : ℝ → ℝ) (M : ℝ → ParamMatrix N)
    (hlam : ∀ t : ℝ, 0 ≤ t → 0 < lam t)
    (hMsymm : ∀ t : ℝ, 0 ≤ t → ∀ v w : EucSpace N,
      inner (𝕜 := ℝ) (M t v) w = inner (𝕜 := ℝ) v (M t w))
    (hMlower : ∀ t : ℝ, 0 ≤ t → ∀ v : EucSpace N, v ≠ 0 →
      lam t * ‖v‖ ^ 2 < inner (𝕜 := ℝ) (M t v) v)
    (hMupper : ∀ t : ℝ, 0 ≤ t → ∀ v : EucSpace N, v ≠ 0 →
      inner (𝕜 := ℝ) (M t v) v < C * lam t * ‖v‖ ^ 2)
    (hint : Filter.Tendsto (fun T : ℝ => ∫ s in (0 : ℝ)..T, lam s)
      Filter.atTop Filter.atTop)
    (x : ℝ → EucSpace N) (K : Set (EucSpace N)) (hK : IsCompact K) (hKU : K ⊆ U)
    (hxK : ∀ t : ℝ, 0 ≤ t → x t ∈ K)
    (hx : ∀ t : ℝ, 0 ≤ t → HasDerivAt x (-(M t (gradient E (x t)))) t) :
    ∃ xstar ∈ U, Filter.Tendsto x Filter.atTop (nhds xstar) ∧
      gradient E xstar = 0 := by
  exact lojasiewicz_modulated_of_local_gradient_inequality N U hU E hE C lam M
    hlam hMsymm hMlower hMupper hint x K hK hKU hxK hx
    fun z hz _ => analytic_gradient_inequality E z (hE z hz)

/-- All inputs of the convergence theorem are simultaneously satisfiable for a
constant energy and trajectory, `M = I`, `λ = 1/2`, `C = 4`; Appendix D.1 of
arXiv:2510.22026v2. -/
example : ∃ z ∈ (Set.univ : Set (EucSpace 1)),
    Filter.Tendsto (fun _ : ℝ => (0 : EucSpace 1)) Filter.atTop (nhds z) ∧
      gradient (fun _ : EucSpace 1 => (0 : ℝ)) z = 0 := by
  apply lojasiewicz_modulated 1 Set.univ isOpen_univ
    (fun _ => 0) analyticOnNhd_const 4 (fun _ => 1 / 2)
    (fun _ => ContinuousLinearMap.id ℝ (EucSpace 1))
    (fun _ _ => by norm_num) (fun _ _ _ _ => rfl)
    (x := fun _ => 0) (K := {0})
  · intro t ht w hw
    rw [ContinuousLinearMap.id_apply, real_inner_self_eq_norm_sq]
    nlinarith [sq_pos_of_pos (norm_pos_iff.mpr hw)]
  · intro t ht w hw
    rw [ContinuousLinearMap.id_apply, real_inner_self_eq_norm_sq]
    nlinarith [sq_pos_of_pos (norm_pos_iff.mpr hw)]
  · have heq : (fun T : ℝ => ∫ _s in (0 : ℝ)..T, (1 / 2 : ℝ)) = fun T : ℝ => T / 2 := by
      ext T
      simp
      ring
    rw [heq]
    exact Filter.tendsto_id.atTop_div_const (by norm_num)
  · exact isCompact_singleton
  · exact Set.subset_univ _
  · intro t ht
    exact Set.mem_singleton 0
  · intro t ht
    simpa using hasDerivAt_const t (0 : EucSpace 1)

end Normalization
end Transformer
