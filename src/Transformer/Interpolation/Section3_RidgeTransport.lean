/-
# Actual single-neuron flow and atomic weak well-posedness

ArXiv:2411.04551v3, §§1, 3.2 and 4, `eq: cauchy.pb`,
`lem: perturbation`, and `eq: neural.ode.sphere`;
Appendix, the flow clauses of `lem: two.balls`.

The complete ridge flow solves the original neural equation with the
explicit single-neuron matrices. Every time slice is Lipschitz and
bijective, and every inactive point stays fixed. Its action on a Dirac
law is the unique original global weak solution from that atom, proved
from the actual weak equation rather than a uniqueness assumption.

The fixed set is stated using the actual affine activation. No identity
between an activation cap and a prescribed source ball is assumed.
-/

import Transformer.Interpolation.Section1_DiracTransport
import Transformer.Interpolation.Section3_BallFlow

open Real MeasureTheory Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The actual single-neuron field is Lipschitz on the unit sphere.
Source: arXiv:2411.04551v3, §1, the Lipschitz well-posedness argument,
and §3.2, the perceptron in `lem: perturbation`. -/
theorem exists_lipschitzOnWith_ridge_field (p u : EucSpace d) (b : ℝ) :
    ∃ K : ℝ≥0, LipschitzOnWith K (fun x : EucSpace d =>
      max (b - inner ℝ x u) 0 • proj d x p) (Sphere d) := by
  obtain ⟨K, hK⟩ := Landscape.GlobalFlow.exists_lipschitzOnWith_closedBall
    (locallyLipschitz_ridgeGlobalField p u b) 1
  refine ⟨K, lipschitzOnWith_iff_norm_sub_le.mpr ?_⟩
  intro x hx y hy
  have hxu := mem_sphere_zero_iff_norm.mp hx
  have hyu := mem_sphere_zero_iff_norm.mp hy
  rw [← ridgeGlobalField_eq_of_unit p u b x hxu, ← ridgeGlobalField_eq_of_unit p u b y hyu]
  exact lipschitzOnWith_iff_norm_sub_le.mp hK (mem_closedBall_zero_iff.mpr hxu.le)
    (mem_closedBall_zero_iff.mpr hyu.le)

/-- Pushing an initial atom along the actual ridge flow gives its
particle's Dirac law. Source: arXiv:2411.04551v3, §1, `eq: ips`. -/
theorem ridgeLaw_diracProb (p u : EucSpace d) (b : ℝ) (x₀ : SSphere d) (t : ℝ) :
    ridgeLaw p u b (Perspective.diracProb d x₀) t =
      Perspective.diracProb d (ridgeFlow p u b x₀ t) := by
  apply Subtype.ext
  change Measure.map (fun x => ridgeFlow p u b x t) (Measure.dirac x₀) =
    Measure.dirac (ridgeFlow p u b x₀ t)
  exact Measure.map_dirac x₀

/-- Every original weak solution from one atom is its actual ridge
particle law, for all real times. Source: arXiv:2411.04551v3,
§1, atomic well-posedness of `eq: cauchy.pb`. -/
theorem cauchyPB_ridge_dirac_unique (i : Fin d) (p u : EucSpace d) (b : ℝ) (x₀ : SSphere d)
    (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => reluRidgeParams i p u b) μ)
    (h0 : μ 0 = Perspective.diracProb d x₀) :
    μ = fun t => Perspective.diracProb d (ridgeFlow p u b x₀ t) := by
  obtain ⟨K, hK⟩ := exists_lipschitzOnWith_ridge_field p u b
  have hF : Continuous (fun x : EucSpace d => max (b - inner ℝ x u) 0 • proj d x p) := by
    unfold proj
    fun_prop
  exact cauchyPB_eq_dirac_transport _ _ hF (fun ρ s x => fullVF_reluRidgeParams i p u b ρ s x)
    K hK μ hμ (ridgeFlow p u b x₀) (hasDerivAt_ridgeFlow p u b x₀)
    (by rw [ridgeFlow_zero]; exact h0)

/-- The explicit ridge controls have exactly one global original weak
solution from any initial atom. Source: arXiv:2411.04551v3, §1,
`eq: cauchy.pb`, restricted to the single-neuron perceptron. -/
theorem exists_unique_cauchyPB_ridge_dirac (i : Fin d) (p u : EucSpace d) (b : ℝ) (x₀ : SSphere d) :
    ∃! μ : ℝ → Perspective.ProbSphere d, μ 0 = Perspective.diracProb d x₀ ∧
      cauchyPB d (fun _ => reluRidgeParams i p u b) μ := by
  refine ⟨ridgeLaw p u b (Perspective.diracProb d x₀),
    ⟨ridgeLaw_zero p u b _, cauchyPB_ridgeLaw i p u b _⟩, fun μ hμ => ?_⟩
  rw [cauchyPB_ridge_dirac_unique i p u b x₀ μ hμ.2 hμ.1]
  funext t
  exact (ridgeLaw_diracProb p u b x₀ t).symm

/-- Every inactive point is fixed by the actual complete flow, including
negative times. Source: arXiv:2411.04551v3, Appendix,
`lem: two.balls`, the identity outside the activated region. -/
theorem ridgeFlow_inactive (p u : EucSpace d) (b : ℝ) (x₀ : SSphere d)
    (hx₀ : b - inner ℝ (x₀ : EucSpace d) u ≤ 0) (t : ℝ) : ridgeFlow p u b x₀ t = x₀ := by
  have he : max (b - inner ℝ (x₀ : EucSpace d) u) 0 • proj d x₀ p = 0 := by
    rw [max_eq_right hx₀, zero_smul]
  have hd (s : ℝ) : HasDerivAt (fun _ : ℝ => (x₀ : EucSpace d))
      (max (b - inner ℝ (x₀ : EucSpace d) u) 0 • proj d x₀ p) s := by
    rw [he]
    exact hasDerivAt_const s _
  have h := ridgeTrajectory_unique p u b (ridgeFlow p u b x₀) (fun _ => x₀)
    (hasDerivAt_ridgeFlow p u b x₀) hd (ridgeFlow_zero p u b x₀)
  exact congrFun h t

/-- The actual ridge flow satisfies the original Lipschitz, bijective
and fixed-point transport clauses off its activation cap.
Source: arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem ridgeFlow_isTransportOutside (p u : EucSpace d) (b T : ℝ) :
    IsTransportOutside d (fun t x => ridgeFlow p u b x t)
      {x : SSphere d | 0 < b - inner ℝ (x : EucSpace d) u} T := by
  intro t _
  refine ⟨exists_lipschitzWith_ridgeFlow p u b t, bijective_ridgeFlow p u b t, fun x hx => ?_⟩
  exact ridgeFlow_inactive p u b x (not_lt.mp hx) t

/-- The explicit ridge matrices realize the original neural vector
field. Source: arXiv:2411.04551v3, §4, `eq: neural.ode.sphere`. -/
theorem neuralVF_reluRidgeParams (i : Fin d) (p u : EucSpace d) (b t : ℝ) (x : SSphere d) :
    neuralVF d (fun _ => (reluRidgeParams i p u b).W) (fun _ => (reluRidgeParams i p u b).U)
      (fun _ => (reluRidgeParams i p u b).b) t x = max (b - inner ℝ (x : EucSpace d) u) 0 • proj d x p := by
  have h := fullVF_reluRidgeParams i p u b (Perspective.diracProb d x) t x
  simpa only [fullVF, neuralVF, reluRidgeParams, zero_apply, zero_add] using h

/-- The actual complete ridge flow satisfies the original integrated
neural-flow equation with the explicit matrices, at every real time.
Source: arXiv:2411.04551v3, §4, `eq: neural.ode.sphere`. -/
theorem isNeuralFlow_ridgeFlow (i : Fin d) (p u : EucSpace d) (b : ℝ) :
    IsNeuralFlow d (fun _ => (reluRidgeParams i p u b).W) (fun _ => (reluRidgeParams i p u b).U)
      (fun _ => (reluRidgeParams i p u b).b) (fun t x => ridgeFlow p u b x t) := by
  intro x t
  simp_rw [neuralVF_reluRidgeParams]
  have hF : Continuous (fun x : EucSpace d => max (b - inner ℝ x u) 0 • proj d x p) := by
    unfold proj
    fun_prop
  have hi : IntervalIntegrable (fun s => max (b - inner ℝ (ridgeFlow p u b x s : EucSpace d) u) 0 •
      proj d (ridgeFlow p u b x s) p) volume 0 t :=
    (hF.comp (continuous_ridgeFlow_time p u b x).subtype_val).intervalIntegrable 0 t
  refine ⟨hi, ?_⟩
  have hFTC := intervalIntegral.integral_eq_sub_of_hasDerivAt (fun s _ => hasDerivAt_ridgeFlow p u b x s) hi
  rw [ridgeFlow_zero] at hFTC
  rw [hFTC]
  module

/-- One neuron has a genuinely activated initial atom and a distinct
inactive point. Its actual weak law realizes the initial-data and weak
solution hypotheses jointly, while the inactive point realizes the fixed
set hypothesis for exactly the same controls. -/
example : ∃ (p u : EucSpace 2) (x₀ z : SSphere 2) (μ : ℝ → Perspective.ProbSphere 2),
    p ≠ 0 ∧ 0 < max (0 - inner ℝ (x₀ : EucSpace 2) u) 0 ∧
    0 - inner ℝ (z : EucSpace 2) u ≤ 0 ∧
    μ 0 = Perspective.diracProb 2 x₀ ∧
    cauchyPB 2 (fun _ => reluRidgeParams (0 : Fin 2) p u 0) μ := by
  let z : SSphere 2 := basePoint 1
  let x₀ : SSphere 2 := antipode 2 z
  let p : EucSpace 2 := EuclideanSpace.single 1 1
  let u : EucSpace 2 := z
  let μ := ridgeLaw p u 0 (Perspective.diracProb 2 x₀)
  have hz : inner ℝ (z : EucSpace 2) (z : EucSpace 2) = 1 := by
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp z.2]
    norm_num
  refine ⟨p, u, x₀, z, μ, ?_, ?_, ?_, ridgeLaw_zero p u 0 _, cauchyPB_ridgeLaw _ _ _ _ _⟩
  · intro hp
    have h := congrArg (fun v : EucSpace 2 => v 1) hp
    norm_num [p] at h
  · change 0 < max (0 - inner ℝ (-(z : EucSpace 2)) (z : EucSpace 2)) 0
    rw [inner_neg_left, hz]
    norm_num
  · change 0 - inner ℝ (z : EucSpace 2) (z : EucSpace 2) ≤ 0
    rw [hz]
    norm_num

end Transformer.Interpolation
