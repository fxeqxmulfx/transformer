/-
# Convergence for a modulated quadratic loss

The complete conclusion of Appendix D.1 of arXiv:2510.22026v2 holds for
quadratic energy without any unproved analytic input. The trajectory's
compact containment follows from energy monotonicity.
-/

import Transformer.Normalization.QuadraticEnergy

namespace Transformer.Normalization

/-- The quadratic-loss case of Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2 is fully proved: a flow `ẋ = -2 M(t)x` with the
source's symmetric matrix bounds and divergent clock converges to zero.
No compact-trajectory hypothesis is needed, since energy dissipation
keeps the trajectory in the closed ball of its initial norm. -/
theorem quadratic_modulated_convergence
    (N : ℕ) (C : ℝ) (lam : ℝ → ℝ) (M : ℝ → ParamMatrix N)
    (hlam : ∀ t : ℝ, 0 ≤ t → 0 < lam t)
    (hMsymm : ∀ t : ℝ, 0 ≤ t → ∀ v w : EucSpace N,
      inner (𝕜 := ℝ) (M t v) w = inner (𝕜 := ℝ) v (M t w))
    (hMlower : ∀ t : ℝ, 0 ≤ t → ∀ v : EucSpace N, v ≠ 0 →
      lam t * ‖v‖ ^ 2 < inner (𝕜 := ℝ) (M t v) v)
    (hMupper : ∀ t : ℝ, 0 ≤ t → ∀ v : EucSpace N, v ≠ 0 →
      inner (𝕜 := ℝ) (M t v) v < C * lam t * ‖v‖ ^ 2)
    (hint : Filter.Tendsto (fun T : ℝ => ∫ s in (0 : ℝ)..T, lam s)
      Filter.atTop Filter.atTop)
    (x : ℝ → EucSpace N)
    (hx : ∀ t : ℝ, 0 ≤ t → HasDerivAt x (-(M t ((2 : ℝ) • x t))) t) :
    Filter.Tendsto x Filter.atTop (nhds 0) := by
  have hflow t (ht : 0 ≤ t) :
      HasDerivAt x (-(M t (gradient (fun y : EucSpace N => ‖y‖ ^ 2) (x t)))) t := by
    simpa only [quadratic_energy_gradient] using hx t ht
  have hdiff t :
      DifferentiableAt ℝ (fun y : EucSpace N => ‖y‖ ^ 2) (x t) :=
    (quadratic_energy_analytic (x t) (Set.mem_univ _)).differentiableAt
  have hanti := modulated_energy_antitoneOn (fun y : EucSpace N => ‖y‖ ^ 2) x M 0
    (fun t _ => hdiff t) hflow (fun t ht => by
      let g := gradient (fun y : EucSpace N => ‖y‖ ^ 2) (x t)
      by_cases hg : g = 0
      · change 0 ≤ inner (𝕜 := ℝ) (M t g) g
        simp [hg]
      · exact (mul_nonneg (hlam t ht).le (sq_nonneg _)).trans (hMlower t ht g hg).le)
  have hxK t (ht : 0 ≤ t) : x t ∈ Metric.closedBall (0 : EucSpace N) ‖x 0‖ := by
    rw [Metric.mem_closedBall, dist_zero_right]
    have henergy := hanti (show (0 : ℝ) ∈ Set.Ici 0 by simp) ht ht
    nlinarith [norm_nonneg (x t), norm_nonneg (x 0)]
  obtain ⟨z, _, hz, hcritical⟩ :=
    lojasiewicz_modulated_of_local_gradient_inequality N Set.univ isOpen_univ
      (fun y : EucSpace N => ‖y‖ ^ 2) quadratic_energy_analytic C lam M
      hlam hMsymm hMlower hMupper hint x (Metric.closedBall 0 ‖x 0‖)
      (isCompact_closedBall _ _) (Set.subset_univ _) hxK hflow (by
        intro z hz hcritical
        rw [quadratic_energy_gradient] at hcritical
        have hz0 : z = 0 := (smul_eq_zero.mp hcritical).resolve_left (by norm_num)
        subst z
        refine ⟨1 / 2, 1, Set.univ, by norm_num, by norm_num, zero_lt_one,
          isOpen_univ, Set.mem_univ _, fun y _ => ?_⟩
        have hzero : ‖(0 : EucSpace N)‖ ^ 2 = (0 : ℝ) := by simp
        rw [hzero, sub_zero, abs_of_nonneg (sq_nonneg ‖y‖), one_mul]
        exact quadratic_energy_gradient_inequality y)
  rw [quadratic_energy_gradient] at hcritical
  have hz0 : z = 0 := (smul_eq_zero.mp hcritical).resolve_left (by norm_num)
  simpa only [hz0] using hz

/-- Identity modulation and the zero trajectory satisfy the full
quadratic convergence hypotheses, with `λ = 1/2`, `C = 4`;
Appendix D.1 of arXiv:2510.22026v2. -/
example : Filter.Tendsto (fun _ : ℝ => (0 : EucSpace 1)) Filter.atTop (nhds 0) := by
  apply quadratic_modulated_convergence 1 4 (fun _ => 1 / 2)
    (fun _ => ContinuousLinearMap.id ℝ (EucSpace 1))
    (fun _ _ => by norm_num) (fun _ _ _ _ => rfl)
  · intro t ht v hv
    rw [ContinuousLinearMap.id_apply, real_inner_self_eq_norm_sq]
    nlinarith [sq_pos_of_pos (norm_pos_iff.mpr hv)]
  · intro t ht v hv
    rw [ContinuousLinearMap.id_apply, real_inner_self_eq_norm_sq]
    nlinarith [sq_pos_of_pos (norm_pos_iff.mpr hv)]
  · have heq : (fun T : ℝ => ∫ _s in (0 : ℝ)..T, (1 / 2 : ℝ)) = fun T : ℝ => T / 2 := by
      ext T
      simp
      ring
    rw [heq]
    exact Filter.tendsto_id.atTop_div_const (by norm_num)
  · intro t ht
    simpa using hasDerivAt_const t (0 : EucSpace 1)

end Transformer.Normalization
