/-
# Analytic modulated gradient convergence in dimensions at most one

The specialization of Appendix D.1, `lem: loj`, of arXiv:2510.22026v2
has a complete proof, including its analytic gradient inequality.
-/

import Transformer.Normalization.LojasiewiczConditional
import Transformer.Normalization.OneDimensionalGradientInequality

namespace Transformer.Normalization

/-- Appendix D.1, `lem: loj`, of arXiv:2510.22026v2 in dimensions
`N ≤ 1`. All hypotheses and the conclusion of the manuscript are
preserved in these dimensions. In particular, arbitrary analytic
energies and degenerate critical points are allowed. The local gradient
inequality is proved from the order of an analytic zero, rather than
assumed as an additional premise. -/
theorem lojasiewicz_modulated_low_dimension
    (N : ℕ) (hN : N ≤ 1) (U : Set (EucSpace N)) (hU : IsOpen U)
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
  exact analytic_gradient_inequality_low_dimension hN E z (hE z hz) hcritical

/-- The dimension-one modulated convergence hypotheses are jointly
satisfiable for constant energy and trajectory,
`M = I`, `λ = 1/2`, `C = 4`. This witnesses the specialization of
Appendix D.1, `lem: loj`, in arXiv:2510.22026v2. -/
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

end Transformer.Normalization
