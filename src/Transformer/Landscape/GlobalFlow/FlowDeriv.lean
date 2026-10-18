/-
# Differentiability of the flow in the initial point

Let `w` be an `IsFlowField` on a finite-dimensional space whose spatial
derivative `A(t, x)` is jointly continuous.

* `‖A‖ ≤ L` (`norm_deriv_le`), and a map whose derivative is jointly
  continuous has a first-order expansion uniform over compact sets of times
  and points (`norm_sub_sub_le_of_uniform`).
* For `s ≤ t`, `x ↦ Φ_{s,t} x` is differentiable, with derivative `J(t)` for
  any solution `J` of the variational equation `J' = A(τ, Φ_{s,τ} x) J`,
  `J(s) = id` (`hasFDerivAt_flow`).  The remainder
  `r = Φ_{s,τ}(x + h) - Φ_{s,τ} x - J(τ) h` is an approximate solution of
  `r' = A r` whose error is `o(‖h‖)`, uniformly in `τ`; Grönwall bounds it.
* In the initial time, `Φ_{s',s} x = x + (s - s') w(s, x) + o(s' - s)`,
  uniformly on balls (`flow_expansion`).

This is the standard theorem on differentiable dependence on initial data,
which Mathlib does not have; arXiv:2305.05465v6, §6, uses it in the duality
proof of uniqueness for `c:wellposedtransformers`.
-/

import Transformer.Landscape.GlobalFlow.Flow

open Set Metric Filter Topology Asymptotics
open scoped NNReal

namespace Transformer
namespace Landscape.GlobalFlow

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  {w : ℝ → E → E} {L : ℝ≥0} {M : ℝ} {A : ℝ → E → E →L[ℝ] E}

/-- The derivative of an `L`-Lipschitz field has norm at most `L`. -/
theorem IsFlowField.norm_deriv_le (hw : IsFlowField w L M)
    (hwA : ∀ t x, HasFDerivAt (w t) (A t x) x) (t : ℝ) (x : E) : ‖A t x‖ ≤ L :=
  (hwA t x).le_of_lipschitz (hw.lipschitz t)

/-- **Uniform differentiability.**  A time-dependent map whose derivative is
jointly continuous is differentiable uniformly over a compact set of times and
a compact convex set of points. -/
theorem norm_sub_sub_le_of_uniform {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    {f : ℝ → E → F} {f' : ℝ → E → E →L[ℝ] F} (hf' : Continuous (Function.uncurry f'))
    (hf : ∀ t x, HasFDerivAt (f t) (f' t x) x) {T : Set ℝ} (hT : IsCompact T) {S : Set E}
    (hS : IsCompact S) (hSc : Convex ℝ S) {ε : ℝ} (hε : 0 < ε) :
    ∃ δ > 0, ∀ t ∈ T, ∀ a ∈ S, ∀ b ∈ S, ‖b - a‖ < δ →
      ‖f t b - f t a - f' t a (b - a)‖ ≤ ε * ‖b - a‖ := by
  obtain ⟨δ, hδ, h⟩ := Metric.uniformContinuousOn_iff.1
    ((hT.prod hS).uniformContinuousOn_of_continuous hf'.continuousOn) ε hε
  refine ⟨δ, hδ, fun t ht a ha b hb hab => ?_⟩
  refine (convex_segment a b).norm_image_sub_le_of_norm_hasFDerivWithin_le' (f := f t)
    (f' := f' t) (φ := f' t a) (fun z _ => (hf t z).hasFDerivWithinAt) (fun z hz => ?_)
    (left_mem_segment ℝ a b) (right_mem_segment ℝ a b)
  rw [← dist_eq_norm]
  refine (h (t, z) ⟨ht, hSc.segment_subset ha hb hz⟩ (t, a) ⟨ht, ha⟩ ?_).le
  rw [Prod.dist_eq, dist_self, max_eq_right dist_nonneg]
  have := segment_subset_closedBall_left a b hz
  rw [mem_closedBall, dist_comm a b] at this
  exact this.trans_lt (by rwa [dist_eq_norm])

variable [FiniteDimensional ℝ E]

/-- **The flow is differentiable in the initial point**, forwards in time, and
its derivative solves the variational equation. -/
theorem IsFlowField.hasFDerivAt_flow (hw : IsFlowField w L M)
    (hA : Continuous (Function.uncurry A)) (hwA : ∀ t x, HasFDerivAt (w t) (A t x) x)
    {s t : ℝ} (hst : s ≤ t) (x : E) {J : ℝ → E →L[ℝ] E}
    (hJs : J s = ContinuousLinearMap.id ℝ E) (hJc : Continuous J)
    (hJ : ∀ τ ∈ Icc s t, HasDerivAt J ((A τ (flow w s τ x)).comp (J τ)) τ) :
    HasFDerivAt (flow w s t) (J t) x := by
  rw [hasFDerivAt_iff_isLittleO_nhds_zero, isLittleO_iff]
  intro c hc
  set Ke := Real.exp (L * (t - s))
  set C₀ := gronwallBound 0 L 1 (t - s)
  set ε := c / (Ke * |C₀| + 1)
  have hε : 0 < ε := by positivity
  set ρ := ‖x‖ + 1 + M * (t - s)
  obtain ⟨δ, hδ, hfd⟩ := norm_sub_sub_le_of_uniform hA hwA (isCompact_Icc (a := s) (b := t))
    (isCompact_closedBall (0 : E) ρ) (convex_closedBall 0 ρ) hε
  have hKe : 0 < Ke := Real.exp_pos _
  filter_upwards [ball_mem_nhds (0 : E) (lt_min one_pos (div_pos hδ hKe))] with h hh
  rw [mem_ball_zero_iff, lt_min_iff] at hh
  set y := fun τ => flow w s τ x
  set z := fun τ => flow w s τ (x + h)
  have hzy : ∀ τ ∈ Icc s t, ‖z τ - y τ‖ ≤ ‖h‖ * Ke := fun τ hτ => by
    have := hw.dist_flow_le_of_le hτ.1 (x + h) x
    rw [dist_eq_norm, dist_eq_norm, add_sub_cancel_left] at this
    exact this.trans (mul_le_mul_of_nonneg_left (Real.exp_le_exp.2
      (mul_le_mul_of_nonneg_left (by linarith [hτ.2]) L.2)) (norm_nonneg _))
  have hmem : ∀ τ ∈ Icc s t, ∀ v : E, ‖v‖ ≤ ‖x‖ + 1 → flow w s τ v ∈ closedBall (0 : E) ρ :=
    fun τ hτ v hv => by
    rw [mem_closedBall_zero_iff]
    have h1 := hw.norm_flow_sub_le s τ v
    have h2 : M * |τ - s| ≤ M * (t - s) := mul_le_mul_of_nonneg_left
      (by rw [abs_of_nonneg (by linarith [hτ.1])]; linarith [hτ.2]) hw.nonneg
    linarith [norm_le_norm_add_norm_sub' (flow w s τ v) v]
  have hxh : ‖x + h‖ ≤ ‖x‖ + 1 := (norm_add_le _ _).trans (by linarith)
  have hv : ∀ τ, LipschitzWith L fun r : E => A τ (y τ) r := fun τ =>
    LipschitzWith.of_dist_le_mul fun a b => by
      rw [dist_eq_norm, dist_eq_norm, ← map_sub]
      exact (A τ (y τ)).le_of_opNorm_le (hw.norm_deriv_le hwA _ _) _
  have key := dist_le_of_approx_trajectories_ODE (v := fun τ r => A τ (y τ) r) (K := L)
    (f := fun τ => z τ - y τ - J τ h)
    (f' := fun τ => w τ (z τ) - w τ (y τ) - (A τ (y τ)).comp (J τ) h)
    (g := fun _ => 0) (g' := fun _ => 0) (εf := ε * (‖h‖ * Ke)) (εg := 0) (δ := 0)
    (a := s) (b := t) hv
    (((hw.continuous_flow_time s _).sub (hw.continuous_flow_time s _)).sub
      (hJc.clm_apply continuous_const)).continuousOn
    (fun τ hτ => by
      have h1 := ((hw.hasDerivAt_flow s τ (x + h)).sub (hw.hasDerivAt_flow s τ x)).sub
        ((hJ τ (Ico_subset_Icc_self hτ)).clm_apply (hasDerivAt_const τ h))
      have h2 : w τ (z τ) - w τ (y τ) - (A τ (y τ)).comp (J τ) h = w τ (flow w s τ (x + h))
          - w τ (flow w s τ x) - ((A τ (flow w s τ x)).comp (J τ) h + J τ 0) := by
        simp [y, z]
      rw [h2]
      exact h1.hasDerivWithinAt)
    (fun τ hτ => by
      have hτ' := Ico_subset_Icc_self hτ
      have he : w τ (z τ) - w τ (y τ) - (A τ (y τ)).comp (J τ) h - A τ (y τ) (z τ - y τ - J τ h)
          = w τ (z τ) - w τ (y τ) - A τ (y τ) (z τ - y τ) := by
        simp only [map_sub, ContinuousLinearMap.comp_apply]; abel
      rw [dist_eq_norm, he]
      refine (hfd τ hτ' _ (hmem τ hτ' x (by linarith)) _ (hmem τ hτ' (x + h) hxh)
        ((hzy τ hτ').trans_lt ?_)).trans (mul_le_mul_of_nonneg_left (hzy τ hτ') hε.le)
      exact (lt_div_iff₀ hKe).1 hh.2)
    continuousOn_const (fun _ _ => hasDerivWithinAt_const _ _ _) (fun _ _ => by simp)
    (by simp [y, z, hw.flow_self, hJs]) t ⟨hst, le_rfl⟩
  have hgb : gronwallBound 0 L (ε * (‖h‖ * Ke) + 0) (t - s) = ‖h‖ * (Ke * (ε * C₀)) := by
    rw [add_zero, show ε * (‖h‖ * Ke) = ‖h‖ * Ke * ε * 1 by ring, gronwallBound_zero_mul]
    ring
  rw [dist_zero_right, hgb] at key
  calc ‖flow w s t (x + h) - flow w s t x - J t h‖ ≤ ‖h‖ * (Ke * (ε * C₀)) := key
    _ ≤ ‖h‖ * c := by
        refine mul_le_mul_of_nonneg_left ?_ (norm_nonneg _)
        calc Ke * (ε * C₀) ≤ Ke * (ε * |C₀|) :=
              mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left (le_abs_self _) hε.le) hKe.le
          _ = c * (Ke * |C₀| / (Ke * |C₀| + 1)) := by simp only [ε]; ring
          _ ≤ c * 1 := mul_le_mul_of_nonneg_left
              ((div_le_one (by positivity)).2 (by linarith)) hc.le
          _ = c := mul_one c
    _ = c * ‖h‖ := mul_comm _ _

/-- **First-order expansion in the initial time**, uniform on balls:
`Φ_{s',s} x = x + (s - s') w(s, x) + o(s' - s)`.  The trajectory from `x`
stays within `M |s - s'|` of `x`, where `w` is uniformly close to `w(s, x)`;
the mean value theorem bounds the rest. -/
theorem IsFlowField.flow_expansion (hw : IsFlowField w L M) (s r : ℝ) {ε : ℝ} (hε : 0 < ε) :
    ∀ᶠ s' in 𝓝 s, ∀ x ∈ closedBall (0 : E) r,
      ‖flow w s' s x - x - (s - s') • w s x‖ ≤ ε * |s - s'| := by
  have hM := hw.nonneg
  obtain ⟨δ, hδ, hwδ⟩ := Metric.uniformContinuousOn_iff.1
    (((isCompact_Icc (a := s - 1) (b := s + 1)).prod (isCompact_closedBall (0 : E) (r + M)))
      |>.uniformContinuousOn_of_continuous hw.continuous.continuousOn) ε hε
  filter_upwards [ball_mem_nhds s (lt_min one_pos (div_pos hδ (by linarith : (0 : ℝ) < M + 1)))]
    with s' hs' x hx
  rw [mem_ball, Real.dist_eq, lt_min_iff, abs_sub_comm] at hs'
  have hf : ∀ σ, HasDerivAt (fun σ => flow w s' σ x - (σ - s') • w s x)
      (w σ (flow w s' σ x) - w s x) σ := fun σ => by
    have := (hw.hasDerivAt_flow s' σ x).fun_sub
      (((hasDerivAt_id σ).sub_const s').smul_const (w s x))
    rwa [one_smul] at this
  have hbound : ∀ σ ∈ uIcc s' s, ‖w σ (flow w s' σ x) - w s x‖ ≤ ε := fun σ hσ => by
    have h1 : |σ - s'| ≤ |s - s'| := abs_sub_left_of_mem_uIcc hσ
    have h2 := abs_lt.1 ((abs_sub_right_of_mem_uIcc hσ).trans_lt hs'.1)
    have h3 : ‖flow w s' σ x - x‖ ≤ M * |s - s'| :=
      (hw.norm_flow_sub_le s' σ x).trans (mul_le_mul_of_nonneg_left h1 hM)
    have h4 : M * |s - s'| < δ := by
      have := (lt_div_iff₀ (by linarith : (0 : ℝ) < M + 1)).1 hs'.2
      nlinarith [abs_nonneg (s - s')]
    have h5 : |s - s'| < δ := lt_of_lt_of_le hs'.2 (div_le_self hδ.le (by linarith))
    rw [← dist_eq_norm]
    refine (hwδ (σ, flow w s' σ x) ⟨⟨by linarith, by linarith⟩, ?_⟩ (s, x)
      ⟨⟨by linarith, by linarith⟩, ?_⟩ ?_).le
    · rw [mem_closedBall_zero_iff] at hx ⊢
      nlinarith [norm_le_norm_add_norm_sub' (flow w s' σ x) x, abs_nonneg (s - s')]
    · rw [mem_closedBall_zero_iff] at hx ⊢
      linarith
    · rw [Prod.dist_eq, Real.dist_eq, dist_eq_norm]
      refine max_lt ((abs_sub_comm σ s ▸ abs_sub_right_of_mem_uIcc hσ).trans_lt h5) ?_
      exact h3.trans_lt h4
  have hmv := (convex_uIcc s' s).norm_image_sub_le_of_norm_deriv_le
    (fun σ _ => (hf σ).differentiableAt) (fun σ hσ => by rw [(hf σ).deriv]; exact hbound σ hσ)
    left_mem_uIcc right_mem_uIcc
  simp only [hw.flow_self, sub_self, zero_smul, sub_zero, Real.norm_eq_abs] at hmv
  rwa [sub_right_comm] at hmv

/-- The hypotheses of the results above are satisfiable: the zero field on `ℝ`, whose
variational equation is solved by the constant `id`, on `[0, 1]`, with `ε = 1`. -/
example : IsFlowField (fun _ _ : ℝ => (0 : ℝ)) 0 0 ∧
    Continuous (Function.uncurry fun (_ _ : ℝ) => (0 : ℝ →L[ℝ] ℝ)) ∧
    (∀ t x : ℝ, HasFDerivAt ((fun _ _ : ℝ => (0 : ℝ)) t) (0 : ℝ →L[ℝ] ℝ) x) ∧
    Continuous (fun _ : ℝ => ContinuousLinearMap.id ℝ ℝ) ∧
    (∀ τ : ℝ, HasDerivAt (fun _ : ℝ => ContinuousLinearMap.id ℝ ℝ)
      ((0 : ℝ →L[ℝ] ℝ).comp (ContinuousLinearMap.id ℝ ℝ)) τ) ∧
    IsCompact (Icc (0 : ℝ) 1) ∧ Convex ℝ (Icc (0 : ℝ) 1) ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 :=
  ⟨⟨continuous_const, fun _ => LipschitzWith.const 0, fun _ _ => by simp⟩, continuous_const,
    fun _ _ => hasFDerivAt_const 0 _, continuous_const,
    fun τ => by simpa using hasDerivAt_const τ (ContinuousLinearMap.id ℝ ℝ),
    isCompact_Icc, convex_Icc 0 1, zero_le_one, one_pos⟩

end Landscape.GlobalFlow
end Transformer
