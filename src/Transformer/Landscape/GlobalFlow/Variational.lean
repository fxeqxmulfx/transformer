module

public import Transformer.Landscape.GlobalFlow.FlowDeriv

/-!
# The variational equation along a flow

Let `w` be an `IsFlowField` on a finite-dimensional space whose spatial
derivative `A(t, x)` is jointly continuous. Along the trajectory
`τ ↦ Φ_{s,τ} x`, the variational equation `J' = A(τ, Φ_{s,τ} x) J`,
`J(s) = id`, is linear, hence not bounded; its solution is the flow of the
field retracted to a ball (`ballRetract`) which, by Grönwall, the solution
never leaves on `[s, s + T]` (`exists_variational`). Its value at a fixed
time is continuous in the initial point (`continuous_variational`), so the
flow is `C¹` forwards in time (`contDiff_flow`).

This is the second half of the standard theorem on differentiable dependence
on initial data, which Mathlib does not have; arXiv:2305.05465v6, §6, uses it
in the duality proof of uniqueness for `c:wellposedtransformers`.
-/

@[expose] public section

open Set Metric Filter Topology
open scoped NNReal

namespace Transformer
namespace Landscape.GlobalFlow

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  {w : ℝ → E → E} {L : ℝ≥0} {M : ℝ} {A : ℝ → E → E →L[ℝ] E}
  [FiniteDimensional ℝ E]

/-- **The variational equation** `J' = A(τ, Φ_{s,τ} x) J`, `J(s) = id`, has a
solution on `[s, s + T]`, with `‖J(τ)‖ ≤ e^{L (τ - s)}`. -/
theorem IsFlowField.exists_variational (hw : IsFlowField w L M)
    (hA : Continuous (Function.uncurry A)) (hwA : ∀ t x, HasFDerivAt (w t) (A t x) x)
    (s : ℝ) (x : E) (T : ℝ) :
    ∃ J : ℝ → E →L[ℝ] E, J s = ContinuousLinearMap.id ℝ E ∧ Continuous J ∧
      ∀ τ ∈ Icc s (s + T), HasDerivAt J ((A τ (flow w s τ x)).comp (J τ)) τ ∧
        ‖J τ‖ ≤ Real.exp (L * (τ - s)) := by
  set ρ := Real.exp (L * T) + 1
  have hρ : 0 < ρ := by positivity
  obtain ⟨G, hGdef⟩ : ∃ G : ℝ → (E →L[ℝ] E) → (E →L[ℝ] E),
      G = fun τ J => (A τ (flow w s τ x)).comp (ballRetract ρ J) := ⟨_, rfl⟩
  have hGapp : ∀ τ J, G τ J = (A τ (flow w s τ x)).comp (ballRetract ρ J) := fun τ J => by
    rw [hGdef]
  have hAc : Continuous fun τ => A τ (flow w s τ x) :=
    hA.comp (continuous_id.prodMk (hw.continuous_flow_time s x))
  have hGle : ∀ τ J, ‖G τ J‖ ≤ L * ‖ballRetract ρ J‖ := fun τ J => by
    rw [hGapp]
    exact (ContinuousLinearMap.opNorm_comp_le _ _).trans
      (mul_le_mul_of_nonneg_right (hw.norm_deriv_le hwA _ _) (norm_nonneg _))
  have hG : IsFlowField G (L * 2) (L * ρ) :=
    { continuous := by
        rw [hGdef]
        exact (hAc.comp continuous_fst).clm_comp
          ((lipschitzWith_ballRetract hρ).continuous.comp continuous_snd)
      lipschitz := fun τ => LipschitzWith.of_dist_le_mul fun J J' => by
        simp only [hGapp, dist_eq_norm, ← ContinuousLinearMap.comp_sub]
        calc _ ≤ ‖A τ (flow w s τ x)‖ * ‖ballRetract ρ J - ballRetract ρ J'‖ :=
              ContinuousLinearMap.opNorm_comp_le _ _
          _ ≤ L * (2 * ‖J - J'‖) := by
              gcongr
              · exact hw.norm_deriv_le hwA _ _
              · simpa [dist_eq_norm] using (lipschitzWith_ballRetract hρ).dist_le_mul J J'
          _ = _ := by push_cast; ring
      bound := fun τ J => (hGle τ J).trans
        (mul_le_mul_of_nonneg_left (norm_ballRetract_le hρ J) L.2) }
  refine ⟨fun τ => flow G s τ (ContinuousLinearMap.id ℝ E), hG.flow_self s _,
    hG.continuous_flow_time s _, fun τ hτ => ?_⟩
  have hbound : ‖flow G s τ (ContinuousLinearMap.id ℝ E)‖ ≤ Real.exp (L * (τ - s)) := by
    have := norm_le_gronwallBound_of_norm_deriv_right_le
      (f := fun τ => flow G s τ (ContinuousLinearMap.id ℝ E)) (δ := 1) (K := L) (ε := 0)
      (a := s) (b := s + T) (hG.continuous_flow_time s _).continuousOn
      (fun τ _ => (hG.hasDerivAt_flow s τ _).hasDerivWithinAt)
      (by rw [hG.flow_self]; exact ContinuousLinearMap.norm_id_le)
      (fun τ _ => by
        rw [add_zero]
        exact (hGle τ _).trans (mul_le_mul_of_nonneg_left (norm_ballRetract_le_norm hρ _) L.2))
      τ hτ
    simpa [gronwallBound_ε0] using this
  have hfix : ballRetract ρ (flow G s τ (ContinuousLinearMap.id ℝ E))
      = flow G s τ (ContinuousLinearMap.id ℝ E) := by
    refine ballRetract_of_norm_le hρ (hbound.trans ?_)
    have : L * (τ - s) ≤ L * T := mul_le_mul_of_nonneg_left (by linarith [hτ.2]) L.2
    linarith [Real.exp_le_exp.2 this]
  refine ⟨?_, hbound⟩
  have hd := hG.hasDerivAt_flow s τ (ContinuousLinearMap.id ℝ E)
  rwa [hGapp, hfix] at hd

/-- **Solutions of the variational equation depend continuously on the
initial point**: if `J x` solves it along `τ ↦ Φ_{s,τ} x` on `[s, t]`, then
`x ↦ J x t` is continuous. The two solutions solve linear equations whose
coefficients `A(τ, Φ_{s,τ} x)` and `A(τ, Φ_{s,τ} x₀)` are uniformly close;
Grönwall bounds the difference. -/
theorem IsFlowField.continuous_variational (hw : IsFlowField w L M)
    (hA : Continuous (Function.uncurry A)) (hwA : ∀ t x, HasFDerivAt (w t) (A t x) x)
    {s t : ℝ} (hst : s ≤ t) {J : E → ℝ → E →L[ℝ] E}
    (hJs : ∀ x, J x s = ContinuousLinearMap.id ℝ E) (hJc : ∀ x, Continuous (J x))
    (hJ : ∀ x, ∀ τ ∈ Icc s t, HasDerivAt (J x) ((A τ (flow w s τ x)).comp (J x τ)) τ ∧
      ‖J x τ‖ ≤ Real.exp (L * (τ - s))) :
    Continuous fun x => J x t := by
  refine continuous_iff_continuousAt.2 fun x₀ => Metric.continuousAt_iff'.2 fun c hc => ?_
  set Ke := Real.exp (L * (t - s))
  set C₀ := gronwallBound 0 L 1 (t - s)
  set η := c / (2 * (Ke * |C₀| + 1))
  have hKe : 0 < Ke := Real.exp_pos _
  have hη : 0 < η := by positivity
  set ρ := ‖x₀‖ + 1 + M * (t - s)
  obtain ⟨δ, hδ, hAδ⟩ := Metric.uniformContinuousOn_iff.1
    ((isCompact_Icc.prod (isCompact_closedBall (0 : E) ρ)).uniformContinuousOn_of_continuous
      hA.continuousOn) η hη
  filter_upwards [ball_mem_nhds x₀ (lt_min one_pos (div_pos hδ hKe))] with x hx
  rw [mem_ball, lt_min_iff] at hx
  have hmem : ∀ τ ∈ Icc s t, ∀ v : E, ‖v‖ ≤ ‖x₀‖ + 1 → flow w s τ v ∈ closedBall (0 : E) ρ :=
    fun τ hτ v hv => by
    rw [mem_closedBall_zero_iff]
    have h1 := hw.norm_flow_sub_le s τ v
    have h2 : M * |τ - s| ≤ M * (t - s) := mul_le_mul_of_nonneg_left
      (by rw [abs_of_nonneg (by linarith [hτ.1])]; linarith [hτ.2]) hw.nonneg
    linarith [norm_le_norm_add_norm_sub' (flow w s τ v) v]
  have hxn : ‖x‖ ≤ ‖x₀‖ + 1 := by
    have := norm_le_norm_add_norm_sub' x x₀
    rw [← dist_eq_norm] at this
    linarith
  have hdist : ∀ τ ∈ Icc s t, dist (flow w s τ x) (flow w s τ x₀) < δ := fun τ hτ => by
    refine (hw.dist_flow_le_of_le hτ.1 x x₀).trans_lt ?_
    calc dist x x₀ * Real.exp (L * (τ - s)) ≤ dist x x₀ * Ke :=
          mul_le_mul_of_nonneg_left (Real.exp_le_exp.2
            (mul_le_mul_of_nonneg_left (by linarith [hτ.2]) L.2)) dist_nonneg
      _ < δ := (lt_div_iff₀ hKe).1 hx.2
  have key := dist_le_of_approx_trajectories_ODE
    (v := fun τ (J' : E →L[ℝ] E) => (A τ (flow w s τ x₀)).comp J') (K := L)
    (f := J x) (f' := fun τ => (A τ (flow w s τ x)).comp (J x τ))
    (g := J x₀) (g' := fun τ => (A τ (flow w s τ x₀)).comp (J x₀ τ))
    (εf := η * Ke) (εg := 0) (δ := 0) (a := s) (b := t)
    (fun τ => LipschitzWith.of_dist_le_mul fun J₁ J₂ => by
      simp only [dist_eq_norm, ← ContinuousLinearMap.comp_sub]
      exact (ContinuousLinearMap.opNorm_comp_le _ _).trans
        (mul_le_mul_of_nonneg_right (hw.norm_deriv_le hwA _ _) (norm_nonneg _)))
    (hJc x).continuousOn
    (fun τ hτ => (hJ x τ (Ico_subset_Icc_self hτ)).1.hasDerivWithinAt)
    (fun τ hτ => by
      have hτ' := Ico_subset_Icc_self hτ
      simp only [dist_eq_norm, ← ContinuousLinearMap.sub_comp]
      refine (ContinuousLinearMap.opNorm_comp_le _ _).trans
        (mul_le_mul ?_ ?_ (norm_nonneg _) hη.le)
      · rw [← dist_eq_norm]
        refine (hAδ (τ, flow w s τ x) ⟨hτ', hmem τ hτ' x hxn⟩ (τ, flow w s τ x₀)
          ⟨hτ', hmem τ hτ' x₀ (by linarith)⟩ ?_).le
        rw [Prod.dist_eq, dist_self, max_eq_right dist_nonneg]
        exact hdist τ hτ'
      · exact (hJ x τ hτ').2.trans (Real.exp_le_exp.2
          (mul_le_mul_of_nonneg_left (by linarith [hτ'.2]) L.2)))
    (hJc x₀).continuousOn
    (fun τ hτ => (hJ x₀ τ (Ico_subset_Icc_self hτ)).1.hasDerivWithinAt)
    (fun τ _ => by simp)
    (by rw [hJs, hJs, dist_self]) t ⟨hst, le_rfl⟩
  have hgb : gronwallBound 0 L (η * Ke + 0) (t - s) = η * Ke * C₀ := by
    rw [add_zero, show η * Ke = η * Ke * 1 by ring, gronwallBound_zero_mul]
    ring
  rw [hgb] at key
  have h0 : 0 ≤ Ke * |C₀| := by positivity
  calc dist (J x t) (J x₀ t) ≤ η * Ke * C₀ := key
    _ ≤ η * (Ke * |C₀|) := by
        rw [mul_assoc]
        exact mul_le_mul_of_nonneg_left
          (mul_le_mul_of_nonneg_left (le_abs_self _) hKe.le) hη.le
    _ = c * (Ke * |C₀| / (2 * (Ke * |C₀| + 1))) := by simp only [η]; ring
    _ < c * 1 := mul_lt_mul_of_pos_left ((div_lt_one (by positivity)).2 (by linarith)) hc
    _ = c := mul_one c

/-- **The flow is `C¹` in the initial point**, forwards in time. -/
theorem IsFlowField.contDiff_flow (hw : IsFlowField w L M)
    (hA : Continuous (Function.uncurry A)) (hwA : ∀ t x, HasFDerivAt (w t) (A t x) x)
    {s t : ℝ} (hst : s ≤ t) : ContDiff ℝ 1 (flow w s t) := by
  choose J hJs hJc hJ using fun x => hw.exists_variational hA hwA s x (t - s)
  rw [add_sub_cancel] at hJ
  have hd : ∀ x, HasFDerivAt (flow w s t) (J x t) x := fun x =>
    hw.hasFDerivAt_flow hA hwA hst x (hJs x) (hJc x) fun τ hτ => (hJ x τ hτ).1
  refine contDiff_one_iff_fderiv.2 ⟨fun x => (hd x).differentiableAt, ?_⟩
  rw [show fderiv ℝ (flow w s t) = fun x => J x t from funext fun x => (hd x).fderiv]
  exact hw.continuous_variational hA hwA hst hJs hJc hJ

/-- The hypotheses of the results above are satisfiable: for the zero field on
`ℝ`, with derivative `0`, on `[0, 1]`, the constant `id` solves the variational
equation within the bound `e^{0 (τ - 0)}`. -/
example : IsFlowField (fun _ _ : ℝ => (0 : ℝ)) 0 0 ∧
    Continuous (Function.uncurry fun (_ _ : ℝ) => (0 : ℝ →L[ℝ] ℝ)) ∧
    (∀ t x : ℝ, HasFDerivAt ((fun _ _ : ℝ => (0 : ℝ)) t) (0 : ℝ →L[ℝ] ℝ) x) ∧ (0 : ℝ) ≤ 1 ∧
    Continuous (fun _ : ℝ => ContinuousLinearMap.id ℝ ℝ) ∧
    ∀ τ : ℝ, HasDerivAt (fun _ : ℝ => ContinuousLinearMap.id ℝ ℝ)
      ((0 : ℝ →L[ℝ] ℝ).comp (ContinuousLinearMap.id ℝ ℝ)) τ ∧
      ‖ContinuousLinearMap.id ℝ ℝ‖ ≤ Real.exp (((0 : ℝ≥0) : ℝ) * (τ - 0)) :=
  ⟨⟨continuous_const, fun _ => LipschitzWith.const 0, fun _ _ => by simp⟩, continuous_const,
    fun _ _ => hasFDerivAt_const 0 _, zero_le_one, continuous_const,
    fun τ => ⟨by simpa using hasDerivAt_const τ (ContinuousLinearMap.id ℝ ℝ),
      by simp⟩⟩

end Landscape.GlobalFlow
end Transformer
