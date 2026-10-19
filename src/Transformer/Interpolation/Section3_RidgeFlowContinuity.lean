module

public import Transformer.Interpolation.Section3_RidgeFlow
public import Mathlib.Analysis.ODE.Gronwall

/-!
# Continuous dependence of the actual one-neuron flow

ArXiv:2411.04551v3, §1, `eq: neural.ode.sphere`, and §3.2,
`lem: perturbation`, Part 2.

On the compact unit ball the complete ambient field has a Lipschitz
bound. Grönwall compares its actual spherical trajectories. Reversing
time supplies the same conclusion for negative times, so every flow
map is Lipschitz in its initial point and hence measurable.

The proved uniform field bound makes every trajectory Lipschitz in
time with one common constant. Combining this estimate with continuous
dependence on the initial point proves joint continuity. These are the
regularity inputs for constructing actual transported weak solutions.
-/

@[expose] public section

open Real Set Filter MeasureTheory
open scoped NNReal Topology

namespace Transformer.Interpolation

/-- Grönwall comparison for two actual spherical trajectories of a field
that is Lipschitz on the unit ball. The whole paths lie in that ball.
Auxiliary for arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`. -/
theorem dist_sphereODE_le_exp {d : ℕ} (F : EucSpace d → EucSpace d) (K : ℝ≥0)
    (hK : LipschitzOnWith K F (Metric.closedBall 0 1))
    (X Y : ℝ → SSphere d)
    (hX : ∀ t : ℝ, HasDerivAt (fun s => (X s : EucSpace d)) (F (X t)) t)
    (hY : ∀ t : ℝ, HasDerivAt (fun s => (Y s : EucSpace d)) (F (Y t)) t)
    (t : ℝ) (ht : 0 ≤ t) :
    dist (X t) (Y t) ≤ dist (X 0) (Y 0) * Real.exp ((K : ℝ) * t) := by
  have hball (x : SSphere d) : (x : EucSpace d) ∈ Metric.closedBall 0 1 := by
    rw [mem_closedBall_zero_iff, mem_sphere_zero_iff_norm.mp x.2]
  have hd (s : ℝ) := (hX s).sub (hY s)
  have hc : Continuous (fun s => (X s : EucSpace d) - (Y s : EucSpace d)) :=
    continuous_iff_continuousAt.mpr fun s => (hd s).continuousAt
  have hb (s : ℝ) : ‖F (X s) - F (Y s)‖ ≤ (K : ℝ) * ‖(X s : EucSpace d) - (Y s : EucSpace d)‖ :=
    lipschitzOnWith_iff_norm_sub_le.mp hK (hball _) (hball _)
  have hG := norm_le_gronwallBound_of_norm_deriv_right_le (a := 0) (b := t)
    (δ := ‖(X 0 : EucSpace d) - (Y 0 : EucSpace d)‖) (K := (K : ℝ)) (ε := 0)
    hc.continuousOn (fun s _ => (hd s).hasDerivWithinAt) le_rfl
    (fun s _ => by simpa only [add_zero] using hb s) t ⟨ht, le_rfl⟩
  change dist (X t : EucSpace d) (Y t : EucSpace d) ≤
    dist (X 0 : EucSpace d) (Y 0 : EucSpace d) * Real.exp ((K : ℝ) * t)
  simpa only [gronwallBound_ε0, sub_zero, ← dist_eq_norm] using hG

/-- Every actual one-neuron flow map is Lipschitz in its original
initial point, including at negative times.
Source: arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`. -/
theorem exists_lipschitzWith_ridgeFlow {d : ℕ} (p u : EucSpace d) (b t : ℝ) :
    ∃ L : ℝ≥0, LipschitzWith L (fun x : SSphere d => ridgeFlow p u b x t) := by
  obtain ⟨K, hK⟩ := Landscape.GlobalFlow.exists_lipschitzOnWith_closedBall
    (locallyLipschitz_ridgeGlobalField p u b) 1
  refine ⟨Real.toNNReal (Real.exp ((K : ℝ) * |t|)), ?_⟩
  rw [lipschitzWith_iff_dist_le_mul]
  intro x y
  rw [Real.coe_toNNReal _ (Real.exp_pos _).le, mul_comm]
  by_cases ht : 0 ≤ t
  · simpa only [abs_of_nonneg ht, ridgeFlow_zero] using dist_sphereODE_le_exp
      (ridgeGlobalField p u b) K hK (ridgeFlow p u b x) (ridgeFlow p u b y)
      (hasDerivAt_ridgeFlow_extension p u b x) (hasDerivAt_ridgeFlow_extension p u b y) t ht
  · have hrev (z : SSphere d) (r : ℝ) :
        HasDerivAt (fun s => (ridgeFlow p u b z (-s) : EucSpace d))
          (-ridgeGlobalField p u b (ridgeFlow p u b z (-r))) r := by
      simpa only [Function.comp_def, neg_smul, one_smul] using
        (hasDerivAt_ridgeFlow_extension p u b z (-r)).scomp r ((hasDerivAt_id r).neg)
    simpa only [neg_neg, neg_zero, ridgeFlow_zero, abs_of_neg (lt_of_not_ge ht)] using
      dist_sphereODE_le_exp (fun z => -ridgeGlobalField p u b z) K hK.neg
        (fun r => ridgeFlow p u b x (-r)) (fun r => ridgeFlow p u b y (-r))
        (hrev x) (hrev y) (-t) (by linarith)

/-- The original flow is continuous in its spherical initial datum.
Source: arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`. -/
theorem continuous_ridgeFlow_initial {d : ℕ} (p u : EucSpace d) (b t : ℝ) :
    Continuous (fun x : SSphere d => ridgeFlow p u b x t) := by
  obtain ⟨L, hL⟩ := exists_lipschitzWith_ridgeFlow p u b t
  exact hL.continuous

/-- Every original flow map is measurable and can push forward a law.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem measurable_ridgeFlow_initial {d : ℕ} (p u : EucSpace d) (b t : ℝ) :
    Measurable (fun x : SSphere d => ridgeFlow p u b x t) :=
  (continuous_ridgeFlow_initial p u b t).measurable

/-- The original network field at the actual neuron matrices has one
uniform bound over all spherical points and all probability laws.
Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem norm_fullVF_reluRidgeParams_le {d : ℕ} (i : Fin d) (p u : EucSpace d) (b : ℝ)
    (ν : Perspective.ProbSphere d) (t : ℝ) (x : SSphere d) :
    ‖fullVF d (fun _ => reluRidgeParams i p u b) ν t x‖ ≤ 4 * (|b| + ‖u‖) * ‖p‖ := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  rw [fullVF_reluRidgeParams, ← ridgeGlobalField_eq_of_unit p u b x hx]
  simpa only [hx, mul_one] using norm_ridgeGlobalField_le p u b x

/-- All actual trajectories share a uniform Lipschitz constant in time.
It is obtained from the proved bound on the original spherical field.
Source: arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`. -/
theorem lipschitzWith_ridgeFlow_time {d : ℕ} (p u : EucSpace d) (b : ℝ) (x₀ : SSphere d) :
    LipschitzWith (Real.toNNReal (4 * (|b| + ‖u‖) * ‖p‖)) (ridgeFlow p u b x₀) := by
  have hC : 0 ≤ 4 * (|b| + ‖u‖) * ‖p‖ := by positivity
  have hLip : LipschitzWith (Real.toNNReal (4 * (|b| + ‖u‖) * ‖p‖))
      (fun s => (ridgeFlow p u b x₀ s : EucSpace d)) := by
    apply lipschitzWith_of_nnnorm_deriv_le (𝕜 := ℝ)
      (fun s => (hasDerivAt_ridgeFlow_extension p u b x₀ s).differentiableAt)
    intro s
    apply NNReal.coe_le_coe.mp
    rw [coe_nnnorm, Real.coe_toNNReal _ hC, (hasDerivAt_ridgeFlow_extension p u b x₀ s).deriv]
    have h := norm_ridgeGlobalField_le p u b (ridgeFlow p u b x₀ s)
    simpa only [mem_sphere_zero_iff_norm.mp (ridgeFlow p u b x₀ s).2, mul_one] using h
  exact hLip.subtype_mk _

/-- The actual one-neuron flow is jointly continuous in time and its
original spherical initial point. This also covers negative times.
Source: arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`. -/
theorem continuous_ridgeFlow {d : ℕ} (p u : EucSpace d) (b : ℝ) :
    Continuous (fun q : ℝ × SSphere d => ridgeFlow p u b q.2 q.1) := by
  apply continuous_iff_continuousAt.mpr
  intro q₀
  let C := 4 * (|b| + ‖u‖) * ‖p‖
  have ht (q : ℝ × SSphere d) :
      dist (ridgeFlow p u b q.2 q.1) (ridgeFlow p u b q.2 q₀.1) ≤ C * dist q.1 q₀.1 := by
    have h := (lipschitzWith_ridgeFlow_time p u b q.2).dist_le_mul q.1 q₀.1
    rw [Real.coe_toNNReal _ (show 0 ≤ 4 * (|b| + ‖u‖) * ‖p‖ by positivity)] at h
    exact h
  have hc : Continuous (fun q : ℝ × SSphere d => C * dist q.1 q₀.1 +
      dist (ridgeFlow p u b q.2 q₀.1) (ridgeFlow p u b q₀.2 q₀.1)) :=
    (continuous_const.mul (continuous_fst.dist continuous_const)).add
      (((continuous_ridgeFlow_initial p u b q₀.1).comp continuous_snd).dist continuous_const)
  have hlim : Tendsto (fun q : ℝ × SSphere d => C * dist q.1 q₀.1 +
      dist (ridgeFlow p u b q.2 q₀.1) (ridgeFlow p u b q₀.2 q₀.1)) (𝓝 q₀) (𝓝 0) := by
    simpa only [dist_self, mul_zero, add_zero] using hc.continuousAt.tendsto (x := q₀)
  apply tendsto_iff_dist_tendsto_zero.mpr
  apply squeeze_zero (fun _ => dist_nonneg) (fun q => ?_) hlim
  exact (dist_triangle _ (ridgeFlow p u b q.2 q₀.1) _).trans (add_le_add (ht q) le_rfl)

/-- Two stationary nonzero unit paths, the zero field on the genuine
ambient unit ball, and a positive evaluation time jointly satisfy all
comparison hypotheses. The same path supplies both trajectories. -/
example :
    LipschitzOnWith 0 (fun _ : EucSpace 1 => (0 : EucSpace 1)) (Metric.closedBall 0 1) ∧
    (∀ t : ℝ, HasDerivAt (fun _ : ℝ => (basePoint 0 : EucSpace 1)) 0 t) ∧
    (0 : ℝ) ≤ 1 := by
  exact ⟨(LipschitzWith.const (0 : EucSpace 1)).lipschitzOnWith,
    fun t => hasDerivAt_const t _, zero_le_one⟩

end Transformer.Interpolation
