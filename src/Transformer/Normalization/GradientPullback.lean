/-
# Local gradient estimates under analytic parametrization

The chain rule controls the gradient of a pulled-back energy. A local
continuous right inverse transfers its gradient estimate back to the
original energy, as required by Appendix D.1 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.ModulatedCritical

open Filter Set

namespace Transformer.Normalization

/-- The gradient of a pulled-back energy is bounded by the derivative norm
of the parametrization times the original gradient norm. Auxiliary chain-rule
estimate for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem gradient_pullback_norm_le {M N : ℕ}
    (E : EucSpace N → ℝ) (phi : EucSpace M → EucSpace N) (y : EucSpace M)
    (hE : DifferentiableAt ℝ E (phi y)) (hphi : DifferentiableAt ℝ phi y) :
    ‖gradient (E ∘ phi) y‖ ≤ ‖fderiv ℝ phi y‖ * ‖gradient E (phi y)‖ := by
  have hnorm {D : ℕ} (f : EucSpace D → ℝ) (x : EucSpace D) :
      ‖gradient f x‖ = ‖fderiv ℝ f x‖ :=
    (InnerProductSpace.toDual ℝ (EucSpace D)).symm.norm_map _
  have hcomp := hE.hasFDerivAt.comp y hphi.hasFDerivAt
  rw [hnorm, hcomp.fderiv, hnorm, mul_comm]
  exact ContinuousLinearMap.opNorm_comp_le _ _

/-- A proved gradient inequality for a pulled-back energy yields a local
estimate using the original gradient at image points. No inverse of the
parametrization is required. Auxiliary estimate for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem local_gradient_inequality_on_analytic_image {M N : ℕ}
    (E : EucSpace N → ℝ) (phi : EucSpace M → EucSpace N) (w : EucSpace M)
    (hE : AnalyticAt ℝ E (phi w)) (hphi : AnalyticAt ℝ phi w)
    (hlocal : ∃ alpha k : ℝ, ∃ V : Set (EucSpace M),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ w ∈ V ∧
        ∀ y ∈ V, |E (phi y) - E (phi w)| ^ alpha ≤ k * ‖gradient (E ∘ phi) y‖) :
    ∃ alpha k : ℝ, ∃ V : Set (EucSpace M),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ w ∈ V ∧
        ∀ y ∈ V, |E (phi y) - E (phi w)| ^ alpha ≤ k * ‖gradient E (phi y)‖ := by
  obtain ⟨alpha, k, V, ha, ha1, hk, hV, hwV, hineq⟩ := hlocal
  let B : ℝ := ‖fderiv ℝ phi w‖ + 1
  have hB : 0 < B := by dsimp [B]; positivity
  have hDB : ∀ᶠ y in nhds w, ‖fderiv ℝ phi y‖ ≤ B := by
    have hcont := (hphi.contDiffAt : ContDiffAt ℝ 1 phi w).continuousAt_fderiv
      (by norm_num)
    have hlt : ‖fderiv ℝ phi w‖ < B := by dsimp [B]; linarith
    exact (hcont.norm.eventually (eventually_lt_nhds hlt)).mono (fun y hy => hy.le)
  have hevent : ∀ᶠ y in nhds w,
      |E (phi y) - E (phi w)| ^ alpha ≤ (k * B) * ‖gradient E (phi y)‖ := by
    filter_upwards [hphi.eventually_analyticAt,
      hphi.continuousAt.tendsto.eventually hE.eventually_analyticAt,
      hDB, hV.mem_nhds hwV] with y hyphi hyE hyB hyV
    have hgrad : ‖gradient (E ∘ phi) y‖ ≤ B * ‖gradient E (phi y)‖ :=
      (gradient_pullback_norm_le E phi y hyE.differentiableAt
        hyphi.differentiableAt).trans
          (mul_le_mul_of_nonneg_right hyB (norm_nonneg _))
    calc
      |E (phi y) - E (phi w)| ^ alpha ≤ k * ‖gradient (E ∘ phi) y‖ := hineq _ hyV
      _ ≤ k * (B * ‖gradient E (phi y)‖) := mul_le_mul_of_nonneg_left hgrad hk.le
      _ = (k * B) * ‖gradient E (phi y)‖ := by ring
  obtain ⟨W, hW, hWo, hwW⟩ := eventually_nhds_iff.mp hevent
  exact ⟨alpha, k * B, W, ha, ha1, mul_pos hk hB, hWo, hwW, hW⟩

/-- An analytic parametrization with a continuous local right inverse
transfers a proved local gradient inequality to the original energy. Only
the forward derivative is bounded; differentiability of the right inverse
is unnecessary. Auxiliary transport result for Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
theorem local_gradient_inequality_of_analytic_parametrization {M N : ℕ}
    (E : EucSpace N → ℝ) (z : EucSpace N) (hE : AnalyticAt ℝ E z)
    (phi : EucSpace M → EucSpace N) (psi : EucSpace N → EucSpace M)
    (w : EucSpace M) (hphi : AnalyticAt ℝ phi w) (hphi0 : phi w = z)
    (hpsi : ContinuousAt psi z) (hpsi0 : psi z = w)
    (hinv : ∀ᶠ y in nhds z, phi (psi y) = y)
    (hlocal : ∃ alpha k : ℝ, ∃ V : Set (EucSpace M),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ w ∈ V ∧
        ∀ y ∈ V, |E (phi y) - E z| ^ alpha ≤ k * ‖gradient (E ∘ phi) y‖) :
    ∃ alpha k : ℝ, ∃ V : Set (EucSpace N),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
        ∀ y ∈ V, |E y - E z| ^ alpha ≤ k * ‖gradient E y‖ := by
  have hEw : AnalyticAt ℝ E (phi w) := by rw [hphi0]; exact hE
  obtain ⟨alpha, k, V, ha, ha1, hk, hV, hwV, hineq⟩ :=
    local_gradient_inequality_on_analytic_image E phi w hEw hphi
      (by simpa only [hphi0] using hlocal)
  have hpsiT : Tendsto psi (nhds z) (nhds w) := by
    simpa only [hpsi0] using hpsi.tendsto
  have hevent : ∀ᶠ y in nhds z,
      |E y - E z| ^ alpha ≤ k * ‖gradient E y‖ := by
    filter_upwards [hpsiT.eventually (hV.mem_nhds hwV), hinv] with y hyV hyinv
    simpa only [hyinv, hphi0] using hineq (psi y) hyV
  obtain ⟨W, hW, hWo, hzW⟩ := eventually_nhds_iff.mp hevent
  exact ⟨alpha, k, W, ha, ha1, hk, hWo, hzW, hW⟩

/-- Identity parametrization and inverse with constant energy satisfy all
transport hypotheses simultaneously; Appendix D.1 of arXiv:2510.22026v2.
Their differentiability also witnesses the chain-rule estimate above. -/
example : ∃ alpha k : ℝ, ∃ V : Set (EucSpace 1),
    0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ (0 : EucSpace 1) ∈ V ∧
      ∀ y ∈ V, |(0 : ℝ)| ^ alpha ≤ k *
        ‖gradient (fun _ : EucSpace 1 => (0 : ℝ)) y‖ := by
  simpa only [sub_self] using local_gradient_inequality_of_analytic_parametrization
    (fun _ : EucSpace 1 => (0 : ℝ)) 0 analyticAt_const id id 0 analyticAt_id rfl
    continuousAt_id rfl (Filter.Eventually.of_forall (fun _ => rfl))
    ⟨1 / 2, 1, Set.univ, by norm_num, by norm_num, zero_lt_one,
      isOpen_univ, Set.mem_univ _, fun y _ => by simp⟩

end Transformer.Normalization
