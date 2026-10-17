/-
# Weak solutions are the measures transported along trajectories

Let `w` be a field continuous in `(t, x)` and bounded uniformly in `x`, locally
uniformly in `t`.

* **Transported measures are solutions.**  If every `t ↦ Y_t x` solves
  `ẏ = w(t, y)`, then `μ_t = (Y_t)_# μ₀` solves `∂_t μ + div(w μ) = 0` weakly,
  against `C¹` test functions with compact support and at every time
  (`weak_of_map`): differentiate `∫ g(Y_t x) dμ₀` under the integral and
  apply the fundamental theorem of calculus.
* **Solutions are transported measures.**  If `w` is an `IsFlowField` whose
  spatial derivative is jointly continuous, a weakly continuous curve
  `t ↦ μ_t` of probability measures, carried near each time by a fixed ball,
  that solves the equation weakly on `[0, T]` is the image of its initial
  measure under the flow: `μ_t = (Φ_{0,t})_# μ_0` for `t ∈ [0, T]`
  (`eq_map_flow`).  The function `F(s) = ∫ g(Φ_{s,t} x) dμ_s` is continuous
  and stationary from the right (`hasDerivWithinAt_integral_comp_flow_zero`),
  so `F(t) = F(0)` for every `g ∈ C_c^∞`, and both measures are carried by one
  ball, where smooth functions with compact support separate probability
  measures (`SmoothApprox.ext_of_forall_contDiff`).

So the linear continuity equation has exactly one such solution, and a
solution of the nonlinear one of arXiv:2305.05465v6, §6, is the Lagrangian
object that its existence proof builds and its stability estimate
`e:w2estimate` compares.
-/

import Transformer.ContinuityEq.Duality

open Set Metric Filter Topology MeasureTheory
open scoped NNReal

namespace Transformer
namespace ContinuityEq

open SmoothApprox GlobalFlow

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [MeasurableSpace E] [BorelSpace E]
  {w : ℝ → E → E} {M : ℝ}

/-- **Measures pushed along trajectories solve the continuity equation.**  If
every `t ↦ Y_t x` solves `ẏ = w(t, y)`, then `μ_t = (Y_t)_# μ₀` satisfies
`∫ g dμ_t = ∫ g dμ_0 + ∫_0^t ∫ Dg(x) w(σ, x) dμ_σ(x) dσ` for every `C¹` test
function `g` with compact support and every time `t`. -/
theorem weak_of_map (hw : Continuous (Function.uncurry w))
    (hM : ∀ t₀, ∃ M, ∀ᶠ t in 𝓝 t₀, ∀ x, ‖w t x‖ ≤ M)
    {Y : ℝ → E → E} (hYc : ∀ t, Continuous (Y t))
    (hY : ∀ t x, HasDerivAt (fun t => Y t x) (w t (Y t x)) t) (μ₀ : Measure E)
    [IsFiniteMeasure μ₀] {g : E → ℝ} (hg : ContDiff ℝ 1 g) (hgc : HasCompactSupport g) (t : ℝ) :
    ∫ x, g x ∂μ₀.map (Y t) = ∫ x, g x ∂μ₀.map (Y 0) +
      ∫ σ in (0 : ℝ)..t, ∫ x, fderiv ℝ g x (w σ x) ∂μ₀.map (Y σ) := by
  have hDg := hg.continuous_fderiv one_ne_zero
  obtain ⟨Cg, hCg⟩ := hDg.bounded_above_of_compact_support (hgc.fderiv (𝕜 := ℝ))
  obtain ⟨C₀, hC₀⟩ := hg.continuous.bounded_above_of_compact_support hgc
  have hwσ : ∀ σ, Continuous (w σ) := fun σ => hw.comp (Continuous.prodMk_right σ)
  have hYt : ∀ x, Continuous fun σ => Y σ x := fun x =>
    continuous_iff_continuousAt.2 fun σ => (hY σ x).continuousAt
  have hbd : ∀ {σ M}, (∀ x, ‖w σ x‖ ≤ M) → ∀ y, ‖fderiv ℝ g y (w σ y)‖ ≤ Cg * M := fun hσ y =>
    ((fderiv ℝ g y).le_opNorm _).trans
      (mul_le_mul (hCg y) (hσ y) (norm_nonneg _) ((norm_nonneg _).trans (hCg y)))
  have hmap : ∀ σ {f : E → ℝ}, Continuous f → ∫ x, f x ∂μ₀.map (Y σ) = ∫ x, f (Y σ x) ∂μ₀ :=
    fun σ f hf => integral_map (hYc σ).aemeasurable hf.aestronglyMeasurable
  have hF'm : ∀ σ, Continuous fun x => fderiv ℝ g (Y σ x) (w σ (Y σ x)) := fun σ =>
    (hDg.comp (hYc σ)).clm_apply ((hwσ σ).comp (hYc σ))
  have hF' : Continuous fun σ => ∫ x, fderiv ℝ g (Y σ x) (w σ (Y σ x)) ∂μ₀ :=
    continuous_iff_continuousAt.2 fun σ₀ => by
      obtain ⟨M, hM⟩ := hM σ₀
      exact continuousAt_of_dominated (bound := fun _ => Cg * M)
        (Eventually.of_forall fun σ => (hF'm σ).aestronglyMeasurable)
        (hM.mono fun σ hσ => Eventually.of_forall fun x => hbd hσ _) (integrable_const _)
        (Eventually.of_forall fun x =>
          ((hDg.comp (hYt x)).clm_apply (hw.comp (continuous_id.prodMk (hYt x)))).continuousAt)
  have hF : ∀ σ, HasDerivAt (fun σ => ∫ x, g (Y σ x) ∂μ₀)
      (∫ x, fderiv ℝ g (Y σ x) (w σ (Y σ x)) ∂μ₀) σ := fun σ => by
    obtain ⟨M, hM⟩ := hM σ
    exact (hasDerivAt_integral_of_dominated_loc_of_deriv_le (μ := μ₀) (x₀ := σ)
      (F := fun σ x => g (Y σ x)) (F' := fun σ x => fderiv ℝ g (Y σ x) (w σ (Y σ x)))
      (bound := fun _ => Cg * M) hM
      (Eventually.of_forall fun σ => (hg.continuous.comp (hYc σ)).aestronglyMeasurable)
      (Integrable.of_bound (hg.continuous.comp (hYc σ)).aestronglyMeasurable C₀
        (Eventually.of_forall fun x => hC₀ _))
      (hF'm σ).aestronglyMeasurable (Eventually.of_forall fun x σ hσ => hbd hσ _)
      (integrable_const _)
      (Eventually.of_forall fun x σ _ =>
        (hg.differentiable one_ne_zero (Y σ x)).hasFDerivAt.comp_hasDerivAt σ (hY σ x))).2
  have e : (fun σ => ∫ x, fderiv ℝ g x (w σ x) ∂μ₀.map (Y σ)) =
      fun σ => ∫ x, fderiv ℝ g (Y σ x) (w σ (Y σ x)) ∂μ₀ :=
    funext fun σ => hmap σ (hDg.clm_apply (hwσ σ))
  rw [e, intervalIntegral.integral_eq_sub_of_hasDerivAt (fun σ _ => hF σ)
    (hF'.intervalIntegrable 0 t), hmap t hg.continuous, hmap 0 hg.continuous]
  ring

variable [FiniteDimensional ℝ E] {L : ℝ≥0}

/-- The image of a measure carried by `B̄(0, R)` under `Φ_{s,t}` is carried by
`B̄(0, R + M |t - s|)`: trajectories move at speed at most `M`. -/
theorem map_flow_closedBall_compl (hw : IsFlowField w L M) {ν : Measure E} {R : ℝ}
    (hν : ν (closedBall 0 R)ᶜ = 0) (s t : ℝ) :
    ν.map (flow w s t) (closedBall 0 (R + M * |t - s|))ᶜ = 0 := by
  rw [Measure.map_apply (hw.continuous_flow s t).measurable measurableSet_closedBall.compl]
  exact measure_mono_null (fun x hx hx' => hx (mem_closedBall_zero_iff.2
    (hw.norm_flow_le (mem_closedBall_zero_iff.1 hx') s t))) hν

/-- **Uniqueness for the linear continuity equation.**  A weak solution on
`[0, T]` with a `C¹` field is the image of its initial measure under the
flow: `μ_t = (Φ_{0,t})_# μ_0`. -/
theorem eq_map_flow (hw : IsFlowField w L M) {A : ℝ → E → E →L[ℝ] E}
    (hA : Continuous (Function.uncurry A)) (hwA : ∀ t x, HasFDerivAt (w t) (A t x) x)
    {μ : ℝ → Measure E} [∀ t, IsProbabilityMeasure (μ t)]
    (hR : ∀ t₀, ∃ R, ∀ᶠ t in 𝓝 t₀, μ t (closedBall 0 R)ᶜ = 0)
    (hμ : ∀ g : E → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
      Continuous fun t => ∫ x, g x ∂μ t)
    {T : ℝ} (hweak : ∀ g : E → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g → ∀ t ∈ Icc 0 T,
      ∫ x, g x ∂μ t = ∫ x, g x ∂μ 0 + ∫ σ in (0 : ℝ)..t, ∫ x, fderiv ℝ g x (w σ x) ∂μ σ)
    {t : ℝ} (ht : t ∈ Icc 0 T) :
    μ t = (μ 0).map (flow w 0 t) := by
  have hmeas := (hw.continuous_flow 0 t).measurable
  have : IsProbabilityMeasure ((μ 0).map (flow w 0 t)) :=
    (Measure.isProbabilityMeasure_map_iff hmeas.aemeasurable).mpr inferInstance
  obtain ⟨R₁, hR₁⟩ := hR t
  obtain ⟨R₀, hR₀⟩ := hR 0
  set R := max R₁ (R₀ + M * |t - 0|)
  refine ext_of_forall_contDiff (R := R) ?_ ?_ fun g hg hgc => ?_
  · exact measure_mono_null (compl_subset_compl.2 (closedBall_subset_closedBall
      (le_max_left _ _))) hR₁.self_of_nhds
  · exact measure_mono_null (compl_subset_compl.2 (closedBall_subset_closedBall
      (le_max_right _ _))) (map_flow_closedBall_compl hw hR₀.self_of_nhds 0 t)
  have hcont : Continuous fun s => ∫ x, g (flow w s t x) ∂μ s :=
    continuous_integral_param hR hμ continuous_id (f := fun s x => g (flow w s t x))
      (hg.continuous.comp (hw.continuous_flow_left t))
  have hconst := constant_of_has_deriv_right_zero hcont.continuousOn
    (fun s hs => hasDerivWithinAt_integral_comp_flow_zero hw hA hwA hR hμ hweak hg hgc hs.1 hs.2
      ht.2) t ⟨ht.1, le_rfl⟩
  simp only [hw.flow_self] at hconst
  rw [hconst, integral_map hmeas.aemeasurable hg.continuous.aestronglyMeasurable]

/-- The hypotheses of the results above are satisfiable: the zero field, with
derivative `0`, whose trajectories `Y t = id` and flow are constant, the
constant curve `δ_0`, carried by `B̄(0, 0)`, which solves its continuity
equation on `[0, 1]`, and the test function `0`. -/
example : IsFlowField (fun (_ : ℝ) (_ : ℝ) => (0 : ℝ)) 0 0 ∧
    Continuous (Function.uncurry fun (_ _ : ℝ) => (0 : ℝ)) ∧
    (∀ t₀ : ℝ, ∃ M : ℝ, ∀ᶠ t in 𝓝 t₀, ∀ x : ℝ, ‖(fun (_ : ℝ) (_ : ℝ) => (0 : ℝ)) t x‖ ≤ M) ∧
    Continuous (fun x : ℝ => x) ∧
    (∀ t x : ℝ, HasDerivAt (fun (_ : ℝ) => x) ((fun (_ : ℝ) (_ : ℝ) => (0 : ℝ)) t x) t) ∧
    Continuous (Function.uncurry fun (_ _ : ℝ) => (0 : ℝ →L[ℝ] ℝ)) ∧
    (∀ t x : ℝ, HasFDerivAt ((fun (_ : ℝ) (_ : ℝ) => (0 : ℝ)) t) (0 : ℝ →L[ℝ] ℝ) x) ∧
    Measure.dirac (0 : ℝ) (closedBall 0 0)ᶜ = 0 ∧
    (∀ t₀ : ℝ, ∃ R : ℝ, ∀ᶠ _ in 𝓝 t₀, Measure.dirac (0 : ℝ) (closedBall 0 R)ᶜ = 0) ∧
    (∀ g : ℝ → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
      Continuous (fun (_ : ℝ) => ∫ x, g x ∂(Measure.dirac (0 : ℝ))) ∧ ∀ t ∈ Icc (0 : ℝ) 1,
        ∫ x, g x ∂(Measure.dirac (0 : ℝ)) = ∫ x, g x ∂(Measure.dirac (0 : ℝ)) +
          ∫ _ in (0 : ℝ)..t, ∫ x, fderiv ℝ g x 0 ∂(Measure.dirac (0 : ℝ))) ∧
    ContDiff ℝ 1 (0 : ℝ → ℝ) ∧ HasCompactSupport (0 : ℝ → ℝ) ∧ (0 : ℝ) ∈ Icc (0 : ℝ) 1 :=
  ⟨⟨continuous_const, fun _ => LipschitzWith.const 0, fun _ _ => by simp⟩, continuous_const,
    fun _ => ⟨0, Eventually.of_forall fun _ _ => by simp⟩, continuous_id,
    fun _ x => hasDerivAt_const _ x, continuous_const, fun _ _ => hasFDerivAt_const 0 _, by simp,
    fun _ => ⟨0, Eventually.of_forall fun _ => by simp⟩,
    fun _ _ _ => ⟨continuous_const, fun _ _ => by simp⟩, contDiff_const, HasCompactSupport.zero,
    ⟨le_rfl, zero_le_one⟩⟩

end ContinuityEq
end Transformer
