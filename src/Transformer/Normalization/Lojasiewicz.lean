/-
# Normalization — The convergence machinery of the proof (App. D of 2510.22026v2)

Two of the three statements the proof of `thm: convergence` rests on:

* `Lemma lem: loj` — the Łojasiewicz convergence theorem generalized to a
  modulated gradient flow `ẋ = -M(t) ∇E(x)`;
* `Proposition sprop: time_change` — a monotone time change turns `eq: NA`
  with speed factors `s_j` into `eq: NA` with `s_j(t(τ)) / t'(τ)`.

The second is the chain rule, proved in `TimeChange`. For the first, all dynamical
steps are proved in `LojasiewiczConditional`; the remaining input is the local
analytic gradient inequality at degenerate critical points of nonconstant
germs in dimensions at least two. The inequality and convergence are
proved in dimensions at most one in `ModulatedScalar`; nondegenerate
critical points in every dimension are covered by
`NondegenerateGradientInequality`. Normal-crossing germs in analytic
coordinates, including multivariable degenerate critical points,
are covered by `NormalCrossingCharts`; finite compact families of such
parametrizations are covered by `CompactMonomialCharts`.
The remaining general case reduces in `MinimumCurveSelection` to one-sided
analytic curve selection in energy-level gradient-minimizer sets. The
accumulation points used for selection are proved critical in
`GradientMinimumCluster`, and the minima satisfy the analytic polar
equations established in `GradientMinimumPolar`.
The general preparation step is proved over real scalars in
`WeierstrassCoordinates`; `EnergyLevelPreparation` gives finite projections
of all nearby energy levels. `RamifiedPolynomialBranches` proves complete
lifting of real polynomial roots in arbitrary degree after a common power
substitution. `AnalyticEquationCurveLifting` retains finite analytic sign
conditions when lifting a prescribed analytic base curve in any dimension.
`EnergyGradientMinimumLifting` constructs analytic gradient-norm minima
against all nearby roots in each distinguished-variable fiber above that
curve, retaining the actual universal comparison on the fiber.
Real division reduces finite analytic fiber conditions to polynomials.
`RealAnalyticGerms` proves Rückert's finite-basis theorem, and analytic values
have monic relations of the prepared degree. `CompactBallGradientQueries` constructs
finite scalar tests comparing minima with every point of the original compact ball.
`ProjectedGradientMinimumLifting` also retains the comparison with every
point of the original source set when an analytic projected curve with
approaching global-minimum witnesses is given.
The remaining geometric step is selecting that base curve in a projected
set; the gradient-minimizer condition quantifies over entire energy fibers.
`AnalyticPlaneCurveSelection` proves selection in arbitrary nonconstant
analytic plane level germs with finite analytic sign conditions.
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

The local analytic gradient inequality at degenerate critical points of
nonconstant germs in dimensions at least two remains unproved. Dimensions
zero and one, including degenerate critical points, and locally constant
energies and nondegenerate critical points in any dimension are proved.
Normal-crossing germs in analytic coordinates with a continuous local
inverse, and germs admitting finite compact monomial charts, are also proved.
The remaining proof reduces to existence of analytic energy-level comparison
curves; the scalar estimates and their transfer are proved.
Everything after that inequality is proved by
`lojasiewicz_modulated_of_local_gradient_inequality`, including compact
uniformization, convergence of the trajectory, and criticality of the limit.

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
  apply lojasiewicz_modulated_of_local_gradient_inequality N U hU E hE C lam M
    hlam hMsymm hMlower hMupper hint x K hK hKU hxK hx
  intro z hz hcritical
  by_cases hconstant : ∀ᶠ y in nhds z, E y = E z
  · exact local_gradient_inequality_of_eventually_constant E z hconstant
  by_cases hN : N ≤ 1
  · exact analytic_gradient_inequality_low_dimension hN E z (hE z hz) hcritical
  by_cases hnondegenerate : ∃ A : EucSpace N ≃L[ℝ] EucSpace N,
      HasFDerivAt (gradient E) A.toContinuousLinearMap z
  · obtain ⟨A, hA⟩ := hnondegenerate
    obtain ⟨k, V, hk, hV, hzV, hineq⟩ :=
      analytic_gradient_inequality_of_nondegenerate E z (hE z hz) A hA hcritical
    exact ⟨1 / 2, k, V, by norm_num, by norm_num, hk, hV, hzV, hineq⟩
  by_cases hprepared : ∃ (D : ℕ) (phi : EucSpace D → EucSpace N)
      (psi : EucSpace N → EucSpace D) (w : EucSpace D) (p : Fin D → ℕ)
      (u : EucSpace D → ℝ), AnalyticAt ℝ phi w ∧ phi w = z ∧
      ContinuousAt psi z ∧ psi z = w ∧ (∀ᶠ y in nhds z, phi (psi y) = y) ∧
      0 < ∑ j, p j ∧ AnalyticAt ℝ u w ∧ u w ≠ 0 ∧ ∀ᶠ y in nhds w,
        E (phi y) - E z = u y * coordinateMonomial p (y - w)
  · obtain ⟨D, phi, psi, w, p, u, hphi, hphi0, hpsi, hpsi0, hinv, hp, hu, hu0, heq⟩ :=
      hprepared
    exact analytic_gradient_inequality_of_normal_crossing_chart E z (hE z hz)
      phi psi w hphi hphi0 hpsi hpsi0 hinv p hp u hu hu0 heq
  by_cases hcharts : HasCompactMonomialCharts E z
  · exact analytic_gradient_inequality_of_compact_monomial_charts E z (hE z hz) hcharts
  apply local_gradient_inequality_of_analytic_comparison_arcs E z (hE z hz).continuousAt
  obtain ⟨r, hr, hballE⟩ := (hE z hz).exists_ball_analyticOnNhd
  have hsubset : Metric.closedBall z (r / 2) ⊆ Metric.ball z r := fun y hy =>
    (Metric.mem_closedBall.mp hy).trans_lt (half_lt_self hr)
  apply analytic_comparison_arcs_of_minimum_curve_selection E z
    (Metric.closedBall z (r / 2)) (isCompact_closedBall _ _)
    (Metric.closedBall_mem_nhds _ (half_pos hr)) (hballE.mono hsubset) hcritical
  intro sign hsign w hw henergy hgradient hacc
  -- The remaining geometric input is one-sided analytic curve selection
  -- in the positive/negative energy-level gradient-minimizer sets.
  -- Compactness, criticality of their accumulation points, energy-level
  -- coverage by the selected curves, and the scalar power bounds are proved.
  sorry

/-- All inputs of the dimension-one convergence theorem are simultaneously
satisfiable for a constant energy and trajectory, `M = I`, `λ = 1/2`,
`C = 4`; Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ z ∈ (Set.univ : Set (EucSpace 1)),
    Filter.Tendsto (fun _ : ℝ => (0 : EucSpace 1)) Filter.atTop (nhds z) ∧
      gradient (fun _ : EucSpace 1 => (0 : ℝ)) z = 0 := by
  apply lojasiewicz_modulated_low_dimension 1 le_rfl Set.univ isOpen_univ
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
