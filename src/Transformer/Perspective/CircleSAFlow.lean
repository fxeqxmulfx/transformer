module

public import Transformer.Perspective.CircleSAGradient
public import Transformer.Perspective.CircleCritical
public import Transformer.Landscape.SaddleAvoidance.MetricAvoidance
public import Transformer.Landscape.SaddleAvoidance.Periodic
public import Transformer.Landscape.SaddleAvoidance.Necessary
public import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace

/-!
# SA on the circle avoids strict saddles

SA in angles is `θ̇ = -G(θ) ∇f(θ)` on `EucSpace n`, with `f = -𝖤_β` and
`G = diag(n²/Z_i)` (`saGain_gradient_apply`), both invariant under the
periods `2π e_k` (`cos_sub_add_period`), which span the space. So `G ∇f` is
bounded and globally Lipschitz (`exists_lipschitzWith_saField`), and the
solutions of SA in angles are the trajectories of
`Landscape.SaddleAvoidance.metricGradientFlow` (`angularSA_iff`): one from every start
(`angularSA_metricGradientFlow`), and no other (`toLp_eq_metricGradientFlow`).
A critical point of `𝖤_β` with a line of positive second derivative is a
strict saddle of `f` (`isStrictSaddle_of_isStrictSaddleTorusEnergy`), so for
Lebesgue-almost every start no solution converges to one
(`ae_not_tendsto_angularSA_of_isStrictSaddle`).

Source: arXiv:2312.10794v5, Appendix A, Lemma `l:nosaddleconv` with Remark
`r:extensiontoSAcase`: SA is a gradient flow for another metric, and the
saddle property does not depend on the metric. The time-one map of
`Landscape.SaddleAvoidance.MetricAvoidance` replaces the center-stable manifold theorem.
-/

@[expose] public section

open Filter MeasureTheory
open scoped BigOperators Topology NNReal
open Real

namespace Transformer
namespace Perspective

open Landscape.SaddleAvoidance Landscape.GlobalFlow

variable (n : ℕ)

/-- **The field of SA in angles is bounded and globally Lipschitz**: `G` and
`f = -𝖤_β` are smooth and invariant under the periods `2π e_k`, which span
the space, so `G ∇f` is too (`exists_lipschitzWith_of_periodic`). -/
theorem exists_lipschitzWith_saField (β : ℝ) :
    ∃ L : ℝ≥0, (∀ ψ, ‖saGain n β ψ (gradient (fun φ => -eucTorusEnergy n β φ) ψ)‖ ≤ L) ∧
      LipschitzWith L fun ψ => saGain n β ψ (gradient (fun φ => -eucTorusEnergy n β φ) ψ) := by
  refine exists_lipschitzWith_of_periodic ((contDiff_saGain n β).clm_apply
    (contDiff_gradient (contDiff_eucTorusEnergy n β).neg)) (span_periods n) ?_
  rintro _ ⟨k, rfl⟩ y
  dsimp only
  rw [gradient_add_eq fun y => by rw [eucTorusEnergy_add_period]]
  simp only [saGain, circlePartition, cos_sub_add_period]

/-- **SA in angles is the flow `ẋ = -G(x) ∇f(x)`** on `EucSpace n`, with
`f = -𝖤_β` and `G = diag(n²/Z_i)`. -/
theorem angularSA_iff (β : ℝ) (hβ : β ≠ 0) (θ : ℝ → Angles n) :
    angularSA n β θ ↔ ∀ t, HasDerivAt (fun s => WithLp.toLp 2 (θ s))
      (-saGain n β (WithLp.toLp 2 (θ t))
        (gradient (fun φ => -eucTorusEnergy n β φ) (WithLp.toLp 2 (θ t)))) t := by
  have key : ∀ t i, (-saGain n β (WithLp.toLp 2 (θ t))
      (gradient (fun φ => -eucTorusEnergy n β φ) (WithLp.toLp 2 (θ t)))).ofLp i =
      -(circlePartition n β (θ t) i)⁻¹ * ∑ j : Idx n,
        Real.exp (β * Real.cos (θ t i - θ t j)) * Real.sin (θ t i - θ t j) := fun t i => by
    rw [WithLp.ofLp_neg, Pi.neg_apply, saGain_gradient_apply n β hβ, neg_mul]
  unfold angularSA
  refine forall_congr' fun t => ?_
  rw [hasDerivAt_toLp_iff]
  simp only [key]

/-- **Existence**: from every start `ψ₀` the flow `ẋ = -G(x) ∇f(x)` gives a
solution of SA in angles. -/
theorem angularSA_metricGradientFlow (β : ℝ) (hβ : β ≠ 0) (ψ₀ : EucSpace n) :
    angularSA n β fun t =>
      (metricGradientFlow (saGain n β) (fun φ => -eucTorusEnergy n β φ) t ψ₀).ofLp := by
  obtain ⟨L, hM, hL⟩ := exists_lipschitzWith_saField n β
  exact (angularSA_iff n β hβ _).2 fun t => hasDerivAt_metricGradientFlow hL hM t ψ₀

/-- **Uniqueness**: every solution of SA in angles is a trajectory of the
flow `ẋ = -G(x) ∇f(x)`. -/
theorem toLp_eq_metricGradientFlow (β : ℝ) (hβ : β ≠ 0) {θ : ℝ → Angles n}
    (hθ : angularSA n β θ) (t : ℝ) :
    WithLp.toLp 2 (θ t) = metricGradientFlow (saGain n β) (fun φ => -eucTorusEnergy n β φ) t
      (WithLp.toLp 2 (θ 0)) := by
  obtain ⟨L, hM, hL⟩ := exists_lipschitzWith_saField n β
  exact ((isFlowField_neg_metricGradient hL hM).flow_eq
    (γ := fun s => WithLp.toLp 2 (θ s)) ((angularSA_iff n β hβ θ).1 hθ) 0 t).symm

/-- **A strict saddle of `𝖤_β` on the torus is a strict saddle of
`f = -𝖤_β`**: at a critical point the gradient vanishes, and along a line
`θ + s v` on which `𝖤_β` has the second derivative `c > 0`,
`⟪∇²f v, v⟫ = -c < 0`, so the Hessian of `f` has a negative eigenvalue
(`isStrictSaddle_of_inner_neg`). -/
theorem isStrictSaddle_of_isStrictSaddleTorusEnergy (β : ℝ) (hβ : β ≠ 0) {z : Angles n}
    (hz : IsCriticalTorusEnergy n β z) (hs : IsStrictSaddleTorusEnergy n β z) :
    IsStrictSaddle (fun φ => -eucTorusEnergy n β φ) (WithLp.toLp 2 z) := by
  obtain ⟨v, c, ⟨f', hf', hc⟩, hcpos⟩ := hs
  set F := eucTorusEnergy n β with hFdef
  set ψ : EucSpace n := WithLp.toLp 2 z
  set w : EucSpace n := WithLp.toLp 2 v
  have hF : ContDiff ℝ 2 F := contDiff_eucTorusEnergy n β
  have hline : ∀ s : ℝ, HasDerivAt (fun u : ℝ => ψ + u • w) w s := fun s => by
    simpa using ((hasDerivAt_id s).smul_const w).const_add ψ
  have hfun : (fun u : ℝ => torusEnergy n β (fun i => z i + u * v i)) =
      F ∘ fun u => ψ + u • w := by
    funext u
    simp only [hFdef, Function.comp_apply, eucTorusEnergy]
    congr 1
  have hf'eq : f' = fun s => inner ℝ (gradient F (ψ + s • w)) w := funext fun s => by
    rw [gradient, InnerProductSpace.toDual_symm_apply]
    refine (hf' s).unique ?_
    rw [hfun]
    exact (hF.differentiable two_ne_zero _).hasFDerivAt.comp_hasDerivAt s (hline s)
  have hH : HasDerivAt (fun s : ℝ => inner ℝ (gradient F (ψ + s • w)) w)
      (inner ℝ (fderiv ℝ (gradient F) ψ w) w) 0 := by
    have hg := ((contDiff_gradient hF).differentiable one_ne_zero ψ).hasFDerivAt
      |>.comp_hasDerivAt_of_eq 0 (hline 0) (by simp)
    simpa using hg.inner ℝ (hasDerivAt_const (0 : ℝ) w)
  rw [hf'eq] at hc
  have hcH := hc.unique hH
  have hgrad : gradient (fun φ => -F φ) = fun x => -gradient F x :=
    funext (gradient_neg_eucTorusEnergy n β)
  refine isStrictSaddle_of_inner_neg hF.neg (h := w) ?_ ?_
  · rw [hgrad, neg_eq_zero]
    refine PiLp.ext fun i => ?_
    rw [gradient_eucTorusEnergy_apply n β hβ]
    exact (hasDerivAt_torusEnergy n β hβ z i).unique (hz i)
  · rw [hgrad, fderiv_fun_neg, neg_apply, inner_neg_left, ← hcH]
    linarith

/-- **SA in angles almost never converges to a strict saddle of `𝖤_β`**: for
Lebesgue-almost every start `θ₀`, no solution of `angularSA` from `θ₀`
converges to a critical point of `𝖤_β` with a line of positive second
derivative. Such a point is a strict saddle of `f = -𝖤_β`, `G` is symmetric
positive definite and `G ∇f` bounded and Lipschitz, so
`ae_not_tendsto_metricGradientFlow_of_isStrictSaddle` applies to the flow,
whose trajectories the solutions are; Lebesgue measure on `ℝⁿ` is that of
`EucSpace n`. -/
theorem ae_not_tendsto_angularSA_of_isStrictSaddle (β : ℝ) (hβ : β ≠ 0) :
    ∀ᵐ θ₀ ∂(volume : Measure (Angles n)), ∀ θ : ℝ → Angles n, angularSA n β θ → θ 0 = θ₀ →
      ∀ z, IsCriticalTorusEnergy n β z → IsStrictSaddleTorusEnergy n β z →
        ¬Tendsto θ atTop (𝓝 z) := by
  obtain ⟨L, hM, hL⟩ := exists_lipschitzWith_saField n β
  have h := ae_not_tendsto_metricGradientFlow_of_isStrictSaddle (volume : Measure (EucSpace n))
    Measure.AbsolutelyContinuous.rfl (contDiff_eucTorusEnergy n β).neg (contDiff_saGain n β) hL hM
    fun _ _ => ⟨isSymmetric_saGain n β _, fun _ hv => inner_saGain_self_pos n β _ hv⟩
  filter_upwards [(PiLp.volume_preserving_toLp (Idx n)).quasiMeasurePreserving.ae h]
    with θ₀ hθ₀ θ hθ h0 z hz hs hlim
  refine hθ₀ _ (isStrictSaddle_of_isStrictSaddleTorusEnergy n β hβ hz hs) ?_
  refine (((PiLp.continuous_toLp 2 _).tendsto z).comp hlim).congr fun t => ?_
  rw [Function.comp_apply, toLp_eq_metricGradientFlow n β hβ hθ t, h0]

/-- The hypotheses of the theorems above are satisfiable: `β = 1 ≠ 0`, the
consensus `θ_i = c` solves `angularSA`, and `θ = (0, π)` is a critical point
of `𝖤_1` with a line of positive second derivative
(`isStrictSaddleTorusEnergy_of_cut_neg`). -/
example (c : ℝ) : (1 : ℝ) ≠ 0 ∧ angularSA n 1 (fun _ _ => c) ∧
    IsCriticalTorusEnergy 2 1 ![0, π] ∧ IsStrictSaddleTorusEnergy 2 1 ![0, π] := by
  refine ⟨one_ne_zero, fun t i => by simpa using hasDerivAt_const t c, fun i => ?_, ?_⟩
  · convert hasDerivAt_torusEnergy 2 1 one_ne_zero ![0, π] i using 1
    fin_cases i <;> simp [Fin.sum_univ_two]
  · refine isStrictSaddleTorusEnergy_of_cut_neg 2 1 one_ne_zero _ {0} ?_
    rw [show ({0} : Finset (Idx 2))ᶜ = {1} by decide, Finset.sum_singleton, Finset.sum_singleton]
    norm_num [exp_pos]

end Perspective
end Transformer
